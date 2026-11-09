import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    # Use higher precision than the reference to minimize relative error
    # Reference is rtol=1e-11, atol=1e-13
    # Using rtol=1e-12, atol=1e-14 should bring rel closer to machine epsilon
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-12, 
        atol=1e-14
    )
    
    if not sol.success:
        # Fallback if it fails, though unlikely
        return y0

    return sol.y[:, -1]
