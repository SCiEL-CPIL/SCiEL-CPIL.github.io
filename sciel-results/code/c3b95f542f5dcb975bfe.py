"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the ODE using the provided function f with the exact solver settings
    specified in the task contract (DOP853, rtol=1e-11, atol=1e-13) to minimize
    relative error to the reference solution.
    """
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use the provided f with the exact solver settings specified in the task contract
    # to minimize relative error to the reference solution.
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback if solve_ivp fails for some reason
        if sol.y.size > 0:
            return sol.y[:, -1].copy()
        return y0_arr.copy()

    return sol.y[:, -1].copy()
# EVOLVE-BLOCK-END
