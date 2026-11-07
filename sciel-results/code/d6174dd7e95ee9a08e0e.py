import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback
        sol = solve_ivp(
            f, 
            t_span, 
        y0, 
        method='RK45', 
        rtol=1e-11, 
        atol=1e-13
        )
        
    if not sol.success:
        raise RuntimeError("Integration failed")
        
    return sol.y[:, -1]
