"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using a high-order adaptive method.
    
    Parameters:
    -----------
    f : callable
        Function f(t, y) returning y' as an array.
    t_span : tuple
        (t_start, t_end)
    y0 : array_like
        Initial conditions.
        
    Returns:
    --------
    y_end : ndarray
        State at t_end.
    """
    deadline = time.time() + 290.0  # safety margin before 330s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Try solve_ivp with DOP853 (12th order) - very accurate and efficient
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 50.0  # reasonable step size
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    if time.time() > deadline:
        # Return best effort - use a simple fixed-step RK4 as fallback
        return _fallback_rk4(f, t_span, y0)
    
    # Try Radau for stiff problems if DOP853 failed
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    if time.time() > deadline:
        return _fallback_rk4(f, t_span, y0)
    
    # Fallback to BDF
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            rtol=1e-12,
            atol=1e-14
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    # Final fallback: fixed-step RK4 with many steps
    return _fallback_rk4(f, t_span, y0)


def _fallback_rk4(f, t_span, y0):
    """Fixed-step RK4 as a last resort."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
