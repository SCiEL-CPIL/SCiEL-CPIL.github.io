import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # The problem is the Van der Pol oscillator with mu=5.
    # It is stiff. Radau is an implicit method suitable for stiff problems.
    # We use tight tolerances to minimize relative error.
    # The score rewards low relative error and low call count.
    # rtol=1e-13 and atol=1e-15 should push rel error below 1e-14.
    
    # Ensure y0 is a float array
    y0_arr = np.array(y0, dtype=float)
    
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='Radau', 
        rtol=1e-13, 
        atol=1e-15
    )
    
    if not sol.success:
        # If Radau fails, try BDF or RK45 with smaller steps as fallback
        # But Radau should work for this problem.
        # For safety, we can try a different method if it fails.
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='BDF', 
            rtol=1e-13, 
            atol=1e-15
        )
        
    return sol.y[:, -1]
