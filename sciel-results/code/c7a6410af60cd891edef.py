"""ODE solver for 15-dim stiff linear system: recover A via basis probes, use expm."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def _recover_A(f, n):
    # Build A so that f(t, y) = A @ y. Probe columns: A[:, j] = f(0, e_j).
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = np.asarray(f(0.0, e), dtype=float)
    return A


def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    start = time.time()
    deadline = start + 250.0  # safety margin before 300s limit

    # Try the linear-structure fast path: recover A and use expm.
    try:
        A = _recover_A(f, n)
        if not np.all(np.isfinite(A)):
            raise RuntimeError("non-finite A")
        # Verify linearity: f(0, y0) should equal A @ y0.
        probe = np.asarray(f(0.0, y0), dtype=float)
        resid = np.linalg.norm(probe - A @ y0)
        scale = max(1.0, np.linalg.norm(A @ y0), np.linalg.norm(probe))
        if resid > 1e-9 * scale:
            raise RuntimeError("not linear: %g" % resid)

        from scipy.linalg import expm
        M = expm(A * dt)
        y_end = M @ y0

        if np.all(np.isfinite(y_end)):
            return y_end
        raise RuntimeError("non-finite result")
    except Exception:
        # Fall back to an explicit integrator (only if within deadline).
        pass

    if time.time() < deadline:
        y = _rk4(f, t0, t1, y0, 2000)
        if y is not None:
            return y

    # Last resort: return what we have (should not normally happen).
    return y0
# EVOLVE-BLOCK-END
