"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using an implicit Radau solver with analytic Jacobian.
    
    Parameters:
    -----------
    f : callable
        The ODE function f(t, y) -> y'
    t_span : tuple
        (t0, t1) integration interval
    y0 : array-like
        Initial condition
    
    Returns:
    --------
    y_end : ndarray
        Final state at t1
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 300s for safety
    start_time = time.time()
    time_limit = 300.0  # seconds
    
    # Analytic Jacobian for the Robertson problem:
    # y' = f(t, y) where y = [y0, y1, y2]
    # f0 = -0.04*y0 + 1e4*y1*y2
    # f1 =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    # f2 =  3e7*y1^2
    #
    # Jacobian J = df/dy:
    # J[0,0] = -0.04
    # J[0,1] = 1e4*y2
    # J[0,2] = 1e4*y1
    #
    # J[1,0] =  0.04
    # J[1,1] = -1e4*y2 - 6e7*y1
    # J[1,2] = -1e4*y1
    #
    # J[2,0] =  0
    # J[2,1] =  6e7*y1
    # J[2,2] =  0
    
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J
    
    # Use Radau method (implicit Runge-Kutta, 3rd order, A-stable)
    # This is ideal for stiff problems like Robertson
    # rtol and atol control accuracy; tighter gives better score but more evaluations
    
    # Strategy: start with tight tolerances, adapt if time runs out
    # Score = -log10(rel_err) - 0.5*log10(nfev)
    # We want high accuracy (low rel_err) and low nfev
    
    # Robertson is known to be stiff; Radau with analytic Jacobian should handle it efficiently
    # Typical nfev for Radau on Robertson is in the hundreds to low thousands
    
    # Try progressively tighter tolerances until we hit the time limit
    # Start with a reasonable tolerance, then refine if time permits
    
    best_result = None
    best_time = 0.0
    
    # Tolerances to try (from loose to tight)
    # Each step should be fast enough that we can try several
    tolerance_configs = [
        (1e-8, 1e-12),   # moderate
        (1e-10, 1e-14),  # tight
        (1e-12, 1e-16),  # very tight
    ]
    
    for rtol, atol in tolerance_configs:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau',
                jac=jac,
                rtol=rtol,
                atol=atol,
                max_step=1.0,
            )
            
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
            
            if sol.success:
                y_end = sol.y[:, -1]
                if best_result is None or elapsed < best_time:
                    best_result = y_end
                    best_time = elapsed
            else:
                # If solver failed, try the next config
                continue
                
        except Exception as e:
            break
    
    # If no successful solve, try a final fallback with looser tolerance
    if best_result is None:
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=jac,
                rtol=1e-6,
                atol=1e-10,
                max_step=1.0,
            )
            if sol.success:
                best_result = sol.y[:, -1]
        except Exception:
            # Last resort: use a simple explicit method with many steps
            h = (t1 - t0) / 10000
            t = t0
            y = y0.copy()
            for _ in range(10000):
                y = y + h * f(t, y)
                t += h
            best_result = y
    
    return best_result
# EVOLVE-BLOCK-END
