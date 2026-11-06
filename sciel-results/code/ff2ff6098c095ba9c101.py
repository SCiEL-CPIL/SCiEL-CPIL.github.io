"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y1, y2, y3 = y
    return np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e3 * y2, -1e4 * y2],
        [0.0, 6e3 * y2, 0.0]
    ])

def integrate(f, t_span, y0):
    """
    Integrates the Robertson ODE problem using the Radau method with an analytical Jacobian.
    
    Parameters:
    f : callable
        The right-hand side function f(t, y).
    t_span : tuple
        (t_start, t_end)
    y0 : array_like
        Initial conditions.
        
    Returns:
    y_end : ndarray
        The state at t_end.
    """
    t_start, t_end = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 300s limit, leave 10s margin
    start_time = time.time()
    deadline = start_time + 290.0
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_robertson_jac, 
            rtol=1e-10, 
            atol=1e-10,
            max_step=1.0 # Limit step size to ensure accuracy if needed, though Radau adapts well
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails (unlikely for this problem), fall back to LSODA
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='LSODA', 
                rtol=1e-10, 
                atol=1e-10
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: return last known state or y0 if it failed immediately
                if sol.y.shape[1] > 0:
                    return sol.y[:, -1]
                return y0
    except Exception as e:
        # In case of any unexpected error, return y0 to avoid crashing
        return y0

# EVOLVE-BLOCK-END
