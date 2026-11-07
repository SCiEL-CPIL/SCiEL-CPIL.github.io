"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate


def integrate(f, t_span, y0):
    # Deadline guard: the evaluator kills after 330s; use a safety margin.
    start_time = time.time()
    deadline = start_time + 300.0  # 300s safety margin

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    # Primary strategy: use scipy's DOP853 with tight tolerances to match the reference.
    # The reference uses rtol=1e-11, atol=1e-13. We use the same to minimize relative error.
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    # Check time; if we're running out, fall back to a fixed-step RK4 with moderate steps.
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Emergency fallback: simple RK4 with 1000 steps
        h = (t1 - t0) / 1000
        t, y = t0, y0.copy()
        for _ in range(1000):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    # Secondary strategy: use DOP853 with even tighter tolerances for a polish pass.
    try:
        sol2 = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
        )
        if sol2.success and np.all(np.isfinite(sol2.y[:, -1])):
            return sol2.y[:, -1]
    except Exception:
        pass

    # Tertiary fallback: RK4 with more steps
    h = (t1 - t0) / 5000
    t, y = t0, y0.copy()
    for _ in range(5000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
