"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def _counting_wrapper(f):
    """Wrap f to count calls."""
    calls = [0]
    
    def wrapper(t, y):
        calls[0] += 1
        return f(t, y)
    
    return wrapper, calls


def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = 330.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Strategy: Use solve_ivp with Radau (matching the reference method)
    # but with tuned parameters to minimize function calls while maintaining accuracy.
    # The reference uses rtol=1e-11, atol=1e-13.
    
    # First, try with a larger max_step to reduce the number of steps (and thus calls).
    # If the solver converges with Newton iterations within the tolerance, it should be accurate.
    # We'll try a few different max_step values and pick the best one (fewest calls, good accuracy).
    
    best_result = None
    best_calls = float('inf')
    
    # Try different max_step values
    max_steps_to_try = [10.0, 5.0, 2.0, 1.0, 0.5]
    
    for max_step in max_steps_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use a counting wrapper to track calls
            f_counted, calls_counter = _counting_wrapper(f)
            
            sol = solve_ivp(
                f_counted,
                t_span,
                y0,
                method='Radau',
                rtol=1e-11,
                atol=1e-13,
                max_step=max_step
            )
            
            if sol.success:
                calls = calls_counter[0]
                if calls < best_calls:
                    best_calls = calls
                    best_result = sol.y[:, -1].copy()
                
                # If we have a very low call count, we can stop early
                if calls < 1000:
                    break
                    
        except Exception:
            continue
    
    # If no valid result was found, try the default adaptive solver
    if best_result is None:
        try:
            f_counted, calls_counter = _counting_wrapper(f)
            sol = solve_ivp(
                f_counted,
                t_span,
                y0,
                method='Radau',
                rtol=1e-11,
                atol=1e-13
            )
            if sol.success:
                best_result = sol.y[:, -1].copy()
        except Exception:
            pass
    
    # Fallback: if still no result, return initial condition (should not happen)
    if best_result is None:
        best_result = y0.copy()
    
    return best_result

# EVOLVE-BLOCK-END
