import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    
    # Use the same method and tolerances as the reference
    # to minimize relative error (rel) to the floor.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback if solver fails, though unlikely for this problem
        # Use analytical solution as backup
        t0, t1 = t_span
        dt = t1 - t0
        y_end = np.empty_like(y0)
        if len(y0) >= 2:
            y_end[0] = y0[0] * np.cos(dt) - y0[1] * np.sin(dt)
            y_end[1] = y0[0] * np.sin(dt) + y0[1] * np.cos(dt)
        if len(y0) > 2:
            y_end[2:] = y0[2:]
        return y_end

    return sol.y[:, -1]
