"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the harmonic oscillator y0'=y1, y1'=-y0 over t_span starting at y0.
    Uses scipy's DOP853 with tight tolerances to match the reference solution,
    driving relative error to the 1e-14 floor.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Deadline guard: return best result before 330s wall-clock limit
    # DOP853 for this simple ODE is extremely fast (< 0.1s typically)
    deadline = time.time() + 280  # safety margin of 50s
    
    # Use DOP853 with matching tolerances to the reference
    # rtol=1e-11, atol=1e-13 as specified in the task
    sol = solve_ivp(
        f,
        t_span,
        y0_arr,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback: if DOP853 fails, try RK45 with tighter tolerances
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='RK45',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
    
    if sol.success and len(sol.t) > 0:
        return sol.y[:, -1]
    
    # Last resort: simple RK4 with many steps
    h = (t1 - t0) / 10000
    t, y = t0, y0_arr.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
        if time.time() > deadline:
            break
    
    return y
# EVOLVE-BLOCK-END
