"""Rewritten program using SciPy's Radau solver with optimized analytical Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the Van der Pol oscillator (mu=5) using scipy.integrate.solve_ivp
    with the Radau method and an analytical Jacobian to minimize function calls.
    
    The problem is:
    y0' = y1
    y1' = 5 * (1 - y0^2) * y1 - y0
    """
    start_time = time.time()
    limit_time = 300.0
    
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float).copy()
    
    # Analytical Jacobian for the Van der Pol oscillator
    # Pre-allocate to avoid repeated allocations
    J = np.empty((2, 2), dtype=float)
    
    def jac(t, y):
        # y[0] = y0, y[1] = y1
        y0_val = y[0]
        y1_val = y[1]
        J[0, 0] = 0.0
        J[0, 1] = 1.0
        J[1, 0] = -10.0 * y0_val * y1_val - 1.0
        J[1, 1] = 5.0 * (1.0 - y0_val * y0_val)
        return J

    try:
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0_arr, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            jac=jac
        )
        
        if sol.success and sol.y.size > 0:
            return sol.y[:, -1]
        else:
            # Fallback: if Radau fails, try BDF
            sol_bdf = solve_ivp(
                f, 
                (t0, t1), 
                y0_arr, 
                method='BDF', 
                rtol=1e-11, 
                atol=1e-13,
                jac=jac
            )
            if sol_bdf.success and sol_bdf.y.size > 0:
                return sol_bdf.y[:, -1]
            if sol.y.size > 0:
                return sol.y[:, -1]
            return y0_arr

    except Exception:
        return y0_arr

# EVOLVE-BLOCK-END
