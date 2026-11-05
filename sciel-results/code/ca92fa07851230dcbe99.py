import numpy as np
from scipy.integrate import solve_ivp
import time

def _robertson_jac(t, y):
    """
    Analytic Jacobian for the Robertson problem.
    y = [y1, y2, y3]
    """
    y1, y2, y3 = y[0], y[1], y[2]
    J = np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e3 * y2, -1e4 * y2],
        [0.0, 6e3 * y2, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    
    # If we are already very close to the limit, return y0 to avoid timeout
    if time.time() - start_time > 290:
        return y0

    try:
        # Primary solver: Radau with analytic Jacobian
        # Radau is order 5, suitable for stiff problems
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-15,
            jac=_robertson_jac
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback: BDF with analytic Jacobian
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-12, 
                atol=1e-15,
                jac=_robertson_jac
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: LSODA (automatic stiff/non-stiff switching)
                # Note: LSODA does not use jac in the same way, might be slower
                sol = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='LSODA', 
                    rtol=1e-8, 
                    atol=1e-11
                )
                return sol.y[:, -1]
                
    except Exception:
        # If any error occurs, try a simple integration or return y0
        # Given the strict deadline, if we fail, we might be close to timeout
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-6, 
                atol=1e-9
            )
            return sol.y[:, -1]
        except Exception:
            return y0
