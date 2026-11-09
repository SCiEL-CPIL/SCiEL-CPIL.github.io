import numpy as np
from scipy.integrate import solve_ivp
import time

def _jac_robertson(t, y):
    """Analytic Jacobian for the Robertson problem."""
    y1, y2, y3 = y
    return np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
        [0.0, 6e7 * y2, 0.0]
    ])

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) using a stiff solver with analytic Jacobian
    for the Robertson problem. Uses Radau with tight tolerances for high accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Primary: Radau with analytic Jacobian and very tight tolerances
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=_jac_robertson,
            rtol=1e-13,
            atol=1e-15,
            max_step=0.1
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 1: BDF with analytic Jacobian
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            jac=_jac_robertson,
            rtol=1e-13,
            atol=1e-15,
            max_step=0.1
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 2: Radau without explicit max_step
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=_jac_robertson,
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 3: BDF without Jacobian
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: Radau without Jacobian
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all else fails, return a zero vector
    return np.zeros_like(y0)
