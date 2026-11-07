"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 with tight tolerances to minimize relative error.
    # rtol=1e-13 and atol=1e-15 should push rel error well below 1e-12,
    # maximizing the -log10(rel) term in the score.
    # DOP853 is an 8th-order method, so it handles tight tolerances efficiently.
    
    try:
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: RK4 with a moderate fixed step size.
    # This is extremely unlikely to be triggered for this smooth ODE.
    steps = 200
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
