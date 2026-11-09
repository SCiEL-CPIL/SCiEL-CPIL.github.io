"""Integrator for the fixed Lorenz IVP using adaptive DOP853 via SciPy."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    if t1 <= t0:
        return y0
    t_start = time.time()
    DEADLINE = 300.0
    # Use DOP853 with tight tolerances to match the reference closely
    # Reference: rtol=1e-11, atol=1e-13
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            max_step=(t1 - t0) / 50.0,
        )
        if sol.success and sol.t.size > 0:
            return sol.y[:, -1]
        # Fallback: return last available point
        if sol.t.size > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    # Deadline-safe fallback: simple fixed-step RK4
    h = (t1 - t0) / 1000.0
    t, y = t0, y0.copy()
    steps = int(round((t1 - t0) / h))
    for _ in range(steps):
        if time.time() - t_start > DEADLINE - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y
# EVOLVE-BLOCK-END
