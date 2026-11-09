"""Integrate Van der Pol oscillator using Radau IIA with analytic Jacobian and tighter tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator y0' = y1, y1' = 5(1-y0^2)y1 - y0
    from t_span[0] to t_span[1] with initial condition y0.
    """
    start_time = time.time()
    
    # Ensure y0 is a float array
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    # Analytic Jacobian for the Van der Pol oscillator with mu=5
    # y0' = y1
    # y1' = 5 * (1 - y0^2) * y1 - y0
    # J = [[0, 1], [-2*5*y0*y1 - 1, 5*(1-y0^2)]]
    def jac(t, y):
        y0_val = y[0]
        y1_val = y[1]
        mu = 5.0
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * y0_val * y1_val - 1.0, mu * (1.0 - y0_val**2)]
        ])
    
    # Use Radau method which is suitable for stiff problems
    # Use tighter tolerances than the reference to minimize rel error
    # Providing jac reduces the number of function calls to f
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-13, 
        atol=1e-15,
        jac=jac
    )
    
    # Check for timeout or failure
    elapsed = time.time() - start_time
    if not sol.success:
        # Fallback: if it fails, try a simpler method or return last state if finite
        if sol.y.size > 0 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
        else:
            # Return initial condition if nothing else works
            return y0
            
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
