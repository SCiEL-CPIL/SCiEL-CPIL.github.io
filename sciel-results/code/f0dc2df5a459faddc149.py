"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use a stiff solver (Radau) with tight tolerances
    # Radau is an implicit Runge-Kutta method, suitable for stiff problems.
    # We aim for high accuracy to minimize the error term in the score.
    # The cost is the number of function evaluations.
    
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-12,
        max_step=1.0
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=1.0
        )
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
