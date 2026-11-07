"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using a high-accuracy method.
    We use scipy.integrate.solve_ivp. First with DOP853 for efficiency (fewer calls),
    then optionally a Radau polish pass for higher accuracy (lower rel).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track elapsed time to stay well within the 330s limit
    start_time = time.time()
    DEADLINE = 300.0  # safety margin before 330s kill
    
    from scipy.integrate import solve_ivp
    
    # Wrap f to ensure it returns a list/array that solve_ivp can handle
    def f_wrapper(t, y):
        result = f(t, y)
        return np.asarray(result, dtype=float)
    
    # Provide an analytic Jacobian for Radau
    def jac(t, y):
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val**2)]
        ])
    
    best_y = y0
    best_calls = float('inf')
    
    # Strategy 1: DOP853 with standard tight tolerances.
    # DOP853 is an explicit 8th-order method. It typically takes larger steps
    # than implicit methods like Radau for the same accuracy, leading to
    # fewer total function calls.
    try:
        sol1 = solve_ivp(
            f_wrapper, 
            t_span, 
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        
        if sol1.success and sol1.y.shape[1] > 0:
            y1 = sol1.y[:, -1]
            if np.all(np.isfinite(y1)):
                best_y = y1
                best_calls = sol1.t.size # Proxy for calls, actually we don't have exact f calls count from solve_ivp directly in an easy way without wrapping, but we can estimate or just use the result.
                # Note: solve_ivp does not directly expose the number of f calls in a simple attribute.
                # However, the scoring uses the number of calls to f.
                # We are wrapping f, but we aren't counting inside the wrapper.
                # The evaluator counts calls to the original f passed to integrate.
                # So we just need to minimize the number of times f is called.
                
                # Check if we have time for a polish pass
                elapsed = time.time() - start_time
                if elapsed < DEADLINE - 30.0:
                    # Strategy 2: Polish with Radau at extremely tight tolerances.
                    # This should reduce the relative error 'rel' significantly.
                    try:
                        sol2 = solve_ivp(
                            f_wrapper, 
                            t_span, 
                            y0,
                            method='Radau',
                            jac=jac,
                            rtol=1e-14,
                            atol=1e-16,
                            dense_output=False
                        )
                        
                        if sol2.success and sol2.y.shape[1] > 0:
                            y2 = sol2.y[:, -1]
                            if np.all(np.isfinite(y2)):
                                # The polish pass likely has higher accuracy (lower rel)
                                # but potentially higher calls.
                                # The score trade-off favors lower rel if calls don't increase too much.
                                # Given the 1e-14 cap on rel gain, driving rel below 1e-14 is the goal.
                                best_y = y2
                    except Exception:
                        pass
    except Exception:
        pass
    
    # If DOP853 failed or wasn't run, try Radau with standard tolerances as a fallback
    if best_y is y0:
        try:
            sol3 = solve_ivp(
                f_wrapper, 
                t_span, 
                y0,
                method='Radau',
                jac=jac,
                rtol=1e-11,
                atol=1e-13,
                dense_output=False
            )
            if sol3.success and sol3.y.shape[1] > 0:
                y3 = sol3.y[:, -1]
                if np.all(np.isfinite(y3)):
                    best_y = y3
        except Exception:
            pass
            
    return best_y
# EVOLVE-BLOCK-END
