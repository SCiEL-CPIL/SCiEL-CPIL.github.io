"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as integrate_scipy

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Robertson problem is stiff. 
    # Using solve_ivp with Radau or BDF is much more efficient than explicit RK4.
    # We need to count function calls. solve_ivp doesn't expose this directly in the result,
    # but we can wrap f to count calls.
    
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Deadline guard: 300s limit, use 280s for safety.
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Try Radau first (3rd order implicit, good for stiff).
    # If it fails or takes too long, try BDF.
    # We want high accuracy. rtol=1e-10, atol=1e-12 is very tight.
    # The score is -log10(rel_err) - 0.5*log10(calls).
    # Higher accuracy (smaller rel_err) increases score.
    # Fewer calls increases score.
    # Radau typically uses fewer steps than BDF for similar accuracy on this problem?
    # Actually, Radau is often preferred for stiff problems where solution is smooth.
    
    methods = ['Radau', 'BDF', 'LSODA']
    best_y = None
    best_score = -np.inf
    
    for method in methods:
        if time.time() > deadline:
            break
            
        try:
            # Use a Jacobian approximation if not provided, but Robertson is small (3x3).
            # solve_ivp will compute Jacobian via finite differences if not provided.
            # This adds overhead but is handled internally.
            
            sol = integrate_scipy.solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method=method,
                rtol=1e-10, 
                atol=1e-12,
                max_step=1.0 # Avoid huge steps initially? Maybe not needed for adaptive.
            )
            
            if not sol.success:
                continue
                
            y_end = sol.y[:, -1]
            
            # Estimate relative error. We don't have the reference, but we can assume
            # that a tighter tolerance yields a better result.
            # However, we need to compare different runs? 
            # The task says "scored by ... against a SciPy reference solution".
            # This implies the evaluator computes the error. We just return y.
            # But the score depends on the number of calls.
            # So we want the method that achieves high accuracy with fewest calls.
            
            # Since we don't know the true error, we assume the tightest tolerance
            # with the fewest calls is best.
            # Radau and BDF with rtol=1e-10 should be extremely accurate.
            # Let's just pick the one that finished successfully with the fewest calls?
            # Or just return the first successful one with very tight tolerance?
            # Given the budget, Radau is usually very efficient for Robertson.
            
            if best_y is None or call_count[0] < best_score:
                 # We are minimizing calls for a fixed high accuracy target.
                 # Actually, we want to maximize -log10(err) - 0.5 log10(calls).
                 # If we assume rtol=1e-10 gives err ~ 1e-10, log10(err) ~ -10.
                 # Score ~ 10 - 0.5 log10(calls).
                 # So minimizing calls is the main lever once accuracy is "sufficiently high".
                 
                 best_y = y_end
                 best_score = call_count[0]
                 
        except Exception:
            continue
            
    if best_y is None:
        # Fallback to a very simple, robust method if all high-order stiff solvers fail
        # (unlikely for Robertson, but safe).
        # Use LSODA with loose tol just to get *something*? No, better to use a fixed step RK4 with many steps?
        # The seed used 2000 steps. Let's try 10000 steps RK4 if all else fails.
        # But we are out of time likely.
        # Let's just return y0 if nothing worked, or try a quick RK4.
        y = y0.copy()
        h = (t1 - t0) / 10000
        t = t0
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                break
        best_y = y

    return best_y
# EVOLVE-BLOCK-END
