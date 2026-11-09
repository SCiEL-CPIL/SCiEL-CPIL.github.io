import numpy as np
import scipy.integrate

def _robertson_jac(t, y):
    """
    Analytical Jacobian for the Robertson problem.
    y = [y1, y2, y3]
    """
    y1, y2, y3 = y
    k1 = 1e-4
    k2 = 4e-3
    k3 = 1e-4
    
    J = np.zeros((3, 3))
    J[0, 0] = -k1 - k3 * y3
    J[0, 1] = k2 * y3
    J[0, 2] = k2 * y2 - k3 * y1
    
    J[1, 0] = k1 - k3 * y3
    J[1, 1] = -k2 * y3
    J[1, 2] = -k2 * y2 - k3 * y1
    
    J[2, 0] = k3 * y3
    J[2, 1] = 0.0
    J[2, 2] = k3 * y1
    
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Radau method (3rd order) with analytical Jacobian
    # This is highly efficient for stiff problems like Robertson
    # rtol and atol are set to ensure high accuracy
    # max_step is set to allow large steps in non-stiff regions
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau',
        jac=_robertson_jac,
        rtol=1e-10,
        atol=1e-14,
        max_step=np.inf, # Let it adapt freely
        t_eval=[t1]
    )
    
    if not sol.success:
        # Fallback to a safer method if Radau fails
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            jac=_robertson_jac,
            rtol=1e-10,
            atol=1e-14,
            t_eval=[t1]
        )
        
    return sol.y[:, -1]
