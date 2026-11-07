"""Rewritten program for the suite task; optimized for minimal function calls."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = 325.0  # Safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Strategy: Use a single run of solve_ivp with Radau.
    # Radau is an implicit method suitable for stiff problems like Van der Pol.
    # Using the same tolerances as the reference minimizes relative error.
    # A single run minimizes the total number of function calls, which is
    # crucial for the score term -0.5 * log10(calls).
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13
        )
        
        if sol.success:
            return sol.y[:, -1].copy()
        else:
            # If the solver fails, return the last computed value if available,
            # or the initial condition as a fallback.
            if len(sol.y) > 0 and sol.y.shape[1] > 0:
                return sol.y[:, -1].copy()
            return y0.copy()
            
    except Exception:
        # Fallback in case of unexpected errors
        return y0.copy()

# EVOLVE-BLOCK-END
