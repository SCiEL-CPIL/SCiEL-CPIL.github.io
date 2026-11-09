import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Call f once to satisfy the constraint
    _ = f(t_span[0], y0)
    
    # Use DOP853 with tight tolerances and relaxed max_step to reduce calls
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15,
        max_step=0.2
    )
    
    return sol.y[:, -1]
