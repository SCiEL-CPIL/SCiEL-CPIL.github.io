import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the initial value problem y' = f(t, y) using an adaptive implicit Runge-Kutta method (Radau).
    This is highly effective for stiff problems like the Robertson problem.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use Radau method which is L-stable and good for stiff problems.
    # Tight tolerances to minimize relative error.
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14,
        max_step=1.0 # Optional: prevent steps from being too large, though adaptive handles it
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails for some reason
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='BDF', 
            rtol=1e-12, 
            atol=1e-14
        )
        
    return sol.y[:, -1]
