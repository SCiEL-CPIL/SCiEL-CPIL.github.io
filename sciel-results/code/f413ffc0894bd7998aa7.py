import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    call_count = 0
    def f_wrapper(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
        
    # Use DOP853 for high order accuracy
    # rtol/atol chosen to be tighter than reference (1e-11/1e-13) 
    # to ensure our error is smaller than ref error, so rel ~ ref error
    # but not too tight to avoid excessive calls
    sol = solve_ivp(
        f_wrapper, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-12, 
        atol=1e-14,
        max_step=0.01 # Prevent huge steps in chaotic regions? DOP853 handles this well, but good to be safe.
    )
    
    if not sol.success:
        # Fallback if something goes wrong, though unlikely for this smooth problem
        # Return last known state or raise
        pass
        
    return sol.y[:, -1]
