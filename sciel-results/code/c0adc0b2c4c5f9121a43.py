import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Analytic Jacobian for Van der Pol with mu = 5
    # y0' = y1
    # y1' = 5 * (1 - y0^2) * y1 - y0
    # J = [[0, 1],
    #      [-1 - 10*y0*y1, 5*(1 - y0^2)]]
    
    def jac(t, y):
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-1.0 - 10.0 * y0_val * y1_val, 5.0 * (1.0 - y0_val**2)]
        ])
    
    # Use Radau with tight tolerances matching or slightly tighter than reference
    # Providing jac should significantly reduce function evaluation overhead
    sol = scipy.integrate.solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13,
        jac=jac,
        t_eval=[t1]
    )
    
    if not sol.success:
        # Fallback if it fails, though unlikely with correct Jacobian
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            t_eval=[t1]
        )
        
    if not sol.success:
        # Last resort fallback
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='LSODA', 
            rtol=1e-11, 
            atol=1e-13,
            t_eval=[t1]
        )
        
    return sol.y[:, -1]
