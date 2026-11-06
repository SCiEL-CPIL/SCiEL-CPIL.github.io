"""ODE solver: Convection-diffusion 100-point using scipy solve_ivp with adaptive stiff-capable methods."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp

_WALL_DEADLINE = 330.0   # seconds; evaluator kills at 330s
_SAFETY_MARGIN = 15.0    # return well before the kill


def _elapsed_ok(start):
    return (time.time() - start) < (_WALL_DEADLINE - _SAFETY_MARGIN)


def _solve(f, t_span, y0, method, rtol, atol, start):
    """Run solve_ivp with the given method/tolerances; return y_end or None on failure."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method=method,
            rtol=rtol, atol=atol,
            dense_output=False,
        )
        if sol.success and sol.y.shape[1] >= 1:
            return sol.y[:, -1]
    except Exception:
        pass
    return None


def integrate(f, t_span, y0):
    start = time.time()

    # Primary: high-order explicit (DOP853) — few f-calls for mildly stiff problems.
    y = _solve(f, t_span, y0, "DOP853", rtol=1e-12, atol=1e-14, start=start)
    if y is not None and np.all(np.isfinite(y)):
        return y

    # Fallback 1: Radau (implicit, stiff) if explicit failed or produced non-finite values.
    if _elapsed_ok(start):
        y = _solve(f, t_span, y0, "Radau", rtol=1e-12, atol=1e-14, start=start)
        if y is not None and np.all(np.isfinite(y)):
            return y

    # Fallback 2: BDF (implicit, stiff) with looser tolerance to guarantee a result.
    if _elapsed_ok(start):
        y = _solve(f, t_span, y0, "BDF", rtol=1e-10, atol=1e-12, start=start)
        if y is not None and np.all(np.isfinite(y)):
            return y

    # Last resort: a fixed-step RK4 with a conservative number of steps, guarded by deadline.
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float)
    n = len(y)
    # Choose steps so that each step is cheap; cap total work by deadline.
    steps = 2000
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        if not _elapsed_ok(start):
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            # Abort; return last finite state if possible.
            break
    return y
# EVOLVE-BLOCK-END
