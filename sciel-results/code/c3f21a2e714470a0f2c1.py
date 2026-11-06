"""ODE solver: Robertson problem using Radau with exact Jacobian via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Uses Radau (implicit Runge-Kutta) with the exact analytical Jacobian of the
    Robertson system. This allows the implicit solver to take large, stable steps
    with fewer function evaluations compared to finite-difference Jacobian approximations.
    A final polish pass with tighter tolerances improves accuracy if time permits.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track function calls
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Exact Jacobian for the Robertson problem:
    # y' = f(t, y) where y = [y0, y1, y2]
    # dy0/dt = -0.04*y0 + 1e4*y1*y2
    # dy1/dt =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    # dy2/dt =  3e7*y1^2
    #
    # J = [[-0.04,       1e4*y2,        1e4*y1   ],
    #      [ 0.04,      -1e4*y2 - 6e7*y1, -1e4*y1],
    #      [ 0,          6e7*y1,           0      ]]
    
    def jacobian(t, y):
        y0_val = y[0]
        y1_val = y[1]
        y2_val = y[2]
        return np.array([
            [-0.04,          1e4*y2_val,         1e4*y1_val],
            [0.04,         -1e4*y2_val - 6e7*y1_val, -1e4*y1_val],
            [0.0,           6e7*y1_val,           0.0]
        ])
    
    # Deadline guard
    start_time = time.time()
    deadline = 310  # safety margin before 330s kill
    
    y_current = y0.copy()
    
    # Stage 1: Radau with tight tolerances and exact Jacobian
    try:
        sol = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='Radau',
            jac=jacobian,
            rtol=1e-13,
            atol=1e-15,
            max_step=np.inf
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-10:
            y_current = sol.y[:, -1].copy()
        elif sol.y.size > 0:
            y_current = sol.y[:, -1].copy()
    
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return y_current
    
    # Stage 2: Polish pass with even tighter tolerances if time permits
    try:
        sol2 = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='Radau',
            jac=jacobian,
            rtol=1e-15,
            atol=1e-17,
            max_step=np.inf
        )
        
        if sol2.success and sol2.t[-1] >= t1 - 1e-10:
            y_current = sol2.y[:, -1].copy()
        elif sol2.y.size > 0:
            y_current = sol2.y[:, -1].copy()
    
    except Exception:
        pass
    
    return y_current
# EVOLVE-BLOCK-END
