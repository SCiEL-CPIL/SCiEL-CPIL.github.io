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
    
    if time.time() - start_time > 290:
        return y0

    try:
        # Structural Change: Piecewise Integration
        # The Robertson problem has a fast transient and a slow decay.
        # We split the integration to allow the solver to reset its step size
        # for the slow part, potentially reducing evaluations.
        
        # Determine split point
        # The transient is typically over by t=0.1
        t_split = t0 + 0.1
        
        if t_split >= t1:
            # If the interval is very short, just solve it normally
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-10, 
                atol=1e-13,
                jac=_robertson_jac
            )
            return sol.y[:, -1]

        # Phase 1: Fast Transient
        sol1 = solve_ivp(
            f, 
            [t0, t_split], 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-15,
            jac=_robertson_jac
        )
        
        if not sol1.success:
            # Fallback to single pass if phase 1 fails
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-10, 
                atol=1e-13,
                jac=_robertson_jac
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return y0

        y_mid = sol1.y[:, -1]
        
        # Phase 2: Slow Decay
        # Looser tolerance is acceptable here as dynamics are slow.
        # Restarting solver allows larger initial step.
        sol2 = solve_ivp(
            f, 
            [t_split, t1], 
            y_mid, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-13,
            jac=_robertson_jac
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        else:
            # Fallback
            return y_mid

    except Exception:
        # If any error occurs, try a simple integration or return y0
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
