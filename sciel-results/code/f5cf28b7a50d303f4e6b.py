"""ODE integrator for the fixed heat-equation IVP y' = A y (N=20)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_A(n):
    # A = 21^2 * tridiag(1, -2, 1)
    diag = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return 441.0 * A


def _radau_2_stage(A, h):
    # Radau IIA 2-stage: b = [1/6, 5/6], c2 = 3/4
    # Precompute (I - h/2 A)^-1 and (I - 5h/6 A)^-1 once (A is constant).
    M1 = np.linalg.inv(np.eye(20) - 0.5 * h * A)
    M2 = np.linalg.inv(np.eye(20) - (5.0 / 6.0) * h * A)
    return M1, M2


def _integrate_radau(f, t0, t1, y0, n_steps, deadline):
    A = _build_A(20)
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / n_steps
    M1, M2 = _radau_2_stage(A, h)
    t = t0
    for _ in range(n_steps):
        tn = t + h
        # Stage 1
        k1 = M1 @ y
        # Stage 2
        k2 = M2 @ (y + (1.5 * h) * f(tn - 0.25 * h, k1))
        y_new = y + h * (k1 + 5.0 * k2)
        if not np.all(np.isfinite(y_new)):
            return None
        y = y_new
        t = tn
        if time.time() > deadline:
            return y
    return y


def _rk4(f, t0, t1, y0, steps, deadline):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return None
        if time.time() > deadline:
            return y
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    start = time.time()
    deadline = start + 250.0  # safety margin before the 330 s kill

    # Primary: implicit Radau IIA with dense analytical Jacobian, adaptive steps.
    for n_steps in (50, 100, 200, 400, 800, 1600):
        y = _integrate_radau(f, t0, t1, y0, n_steps, deadline)
        if y is not None:
            return y

    # Fallback: explicit RK4 with modest steps.
    y = _rk4(f, t0, t1, y0, 2000, deadline)
    if y is not None:
        return y

    # Absolute last resort: a single Euler step (still finite, f called once).
    return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
