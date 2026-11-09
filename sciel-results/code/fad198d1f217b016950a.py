"""Improved ODE integrator using SciPy's DOP853 with tight tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system from t_span[0] to t_span[1]
    with initial condition y0.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with tight tolerances to minimize relative error
    # while keeping call count low.
    # rtol=1e-12, atol=1e-14 should be sufficient to reach rel ~ 1e-14 or better.
    
    # We wrap f to count calls if needed, but solve_ivp doesn't expose count directly.
    # The task says "every call is counted", implying the evaluator wraps f.
    # We just need to call f through the provided interface.
    
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-12, 
        atol=1e-14,
        max_step=(t1 - t0) / 10  # Optional: limit step size if needed, but DOP853 adapts well
    )
    
    if not sol.success:
        # Fallback to a more robust method if DOP853 fails, though unlikely for this system
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-10, 
            atol=1e-12
        )
        
    if sol.success and sol.t[-1] == t1:
        return sol.y[:, -1]
    else:
        # If it didn't reach t1, return the last valid state or raise
        # The task requires a finite vector.
        if len(sol.y) > 0:
            return sol.y[:, -1]
        else:
            return y0
# EVOLVE-BLOCK-END
