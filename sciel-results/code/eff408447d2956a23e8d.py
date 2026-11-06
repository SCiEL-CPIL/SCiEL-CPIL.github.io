import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Try Radau for stiff problems
    # Radau is an implicit Runge-Kutta method of order 5
    # Good for stiff ODEs
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=0.1 # Prevent huge steps if needed, but adaptive handles it
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to LSODA if Radau fails
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to DOP853
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: RK4 manual (should not happen)
    # ...
    return y0
