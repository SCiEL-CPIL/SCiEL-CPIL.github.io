import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses an implicit Runge-Kutta method (Radau) suitable for stiff systems.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use Radau for stiff systems, high accuracy
    # Tight tolerances to minimize relative error
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-12, 
            atol=1e-14
        )
        
    return sol.y[:, -1]
