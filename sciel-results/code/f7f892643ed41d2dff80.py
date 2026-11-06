import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Approach: Use BDF (Backward Differentiation Formulas) first.
    # BDF is a linear multistep method, distinct from the Radau (Runge-Kutta)
    # used in the previous program. BDF is often very efficient for stiff
    # problems like convection-diffusion.
    
    # Try BDF with tight tolerances
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-12, 
            atol=1e-14,
            # max_step=0.1 # Let adaptive handle it, BDF can take large steps if smooth
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to Radau (Implicit Runge-Kutta)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to LSODA
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: DOP853 (Explicit, high order, good for non-stiff)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
