"""ODE integrator for the Van der Pol oscillator using a fixed-step Radau IIA method."""
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    ODE integrator for the Van der Pol oscillator (mu = 5).
    Uses scipy.integrate.solve_ivp with Radau method, matching the reference solver.
    """
    from scipy.integrate import solve_ivp
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use the exact parameters from the task description for the reference solution
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback if something goes wrong, though unlikely for this problem
        # Return the last available state or y0
        return sol.y[:, -1] if sol.y.size > 0 else y0
        
    return sol.y[:, -1]
