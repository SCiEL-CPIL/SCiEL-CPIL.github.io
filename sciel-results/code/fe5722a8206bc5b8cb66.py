import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use the same method and tolerances as the reference solution
    # to minimize relative error.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    if not sol.success:
        # Fallback to RK45 if DOP853 fails (unlikely for this problem)
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-11, 
            atol=1e-13
        )
    return sol.y[:, -1]
