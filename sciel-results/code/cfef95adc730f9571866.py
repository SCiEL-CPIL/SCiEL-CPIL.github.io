"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: 120s limit, use 115s as safety margin
    start_time = time.time()
    deadline = start_time + 115.0
    
    x0 = float(x0)
    
    def check_deadline():
        return time.time() > deadline - 5.0
    
    # Strategy:
    # 1. Global/Local Search: Probe f at points spreading out from x0 geometrically.
    #    This helps if x0 is far from the root or if the root is in a different "basin".
    #    We look for a sign change or a point where |f| is small.
    # 2. High-Order Local Refinement: Once a good candidate (sign change bracket or small |f|)
    #    is found, use a robust high-order method (like Halley's or a modified Newton with 
    #    finite differences) to converge to |f(x)| <= 1e-8.
    #    
    #    Why this is different:
    #    - Current approach assumes the root is close to x0 and fits a local polynomial.
    #    - If x0 is far, the local fit is garbage.
    #    - If the polynomial is high degree, a local fit of degree 5 might miss the root 
    #      if the curvature is high or if there are multiple close roots.
    #    - Probing spreads risk and ensures we find *a* root, not just *the* root near x0 
    #      (though the task implies a unique relevant root or x0 is a good start).
    #    - Actually, the task says "x0 is a float starting point". It doesn't guarantee 
    #      proximity. However, usually in these tasks, x0 is somewhat close. 
    #      The "multi-scale" approach is more robust to x0 being off.
    
    # Phase 1: Search for a bracket or a good starting point
    # We will evaluate f at x0 and then expand outward.
    # Let's try a few scales: 1, 2, 4, 8, ...
    # We keep the point with the smallest |f| as our best candidate.
    
    best_x = x0
    best_f = f(x0)
    best_abs_f = abs(best_f)
    
    # If already good enough
    if best_abs_f <= 1e-8:
        return best_x
    
    # Expand search
    # We'll check x0 + delta and x0 - delta for increasing delta
    # Deltas: 1, 2, 4, 8, 16, 32, 64, 128
    # Limit total calls to ~20 in this phase to save budget for refinement.
    
    deltas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    points_checked = {x0}
    
    for d in deltas:
        if check_deadline():
            break
            
        for sign in [1, -1]:
            x_try = x0 + sign * d
            if x_try in points_checked:
                continue
            points_checked.add(x_try)
            
            try:
                f_try = f(x_try)
                if not np.isfinite(f_try):
                    continue
                abs_f_try = abs(f_try)
                
                if abs_f_try < best_abs_f:
                    best_abs_f = abs_f_try
                    best_x = x_try
                    best_f = f_try
                    
                    if best_abs_f <= 1e-8:
                        return best_x
                        
            except Exception:
                pass
    
    # Now we have a best_x. 
    # If best_x is still far from the root (high |f|), we might need a more robust method.
    # However, for smooth polynomials, if we haven't found a sign change, we might be 
    # in a region where f doesn't cross zero (e.g. local min > 0). But the task guarantees 
    # a root exists. 
    # 
    # Let's check if we can form a bracket.
    # We have evaluated f at x0, x0+1, x0-1, etc.
    # Let's store the evaluations in a list to check for sign changes.
    
    evals = [(x0, best_f if best_x == x0 else f(x0))] # Re-evaluating x0 if it wasn't best? No, we have it.
    # Actually, let's just re-evaluate or keep track properly. 
    # For simplicity and robustness, let's just use the best_x found and run a robust 
    # local solver from there.
    
    # Phase 2: Robust Local Refinement
    # Use a method that converges fast but is robust.
    # Bisection is slow. Newton is fast but can diverge.
    # Secant is good.
    # Let's use a modified Newton with damping and finite difference derivative.
    # Or better: Use the "Illinois" algorithm or a simple Bisection-Newton hybrid.
    
    # Given the polynomial nature, Newton with a good initial guess (best_x) should work.
    # To be safe, we'll use a damped Newton step.
    
    x = best_x
    fx = best_f
    
    # If fx is still large, maybe we should try to find a bracket first?
    # If no sign change was found in the search phase, it's possible the root is 
    # further away or the function is monotonic and we missed it.
    # But the geometric search covers a wide range.
    
    # Let's try up to 50 Newton steps.
    for i in range(50):
        if check_deadline():
            break
            
        if abs(fx) <= 1e-8:
            return x
        
        # Estimate derivative
        # Central difference is more accurate
        h = 1e-6 * max(1.0, abs(x))
        try:
            f_plus = f(x + h)
            f_minus = f(x - h)
            df = (f_plus - f_minus) / (2 * h)
        except Exception:
            break
            
        if df == 0 or not np.isfinite(df):
            # If derivative is zero or invalid, perturb x slightly
            x += 1e-4
            continue
            
        # Newton step
        step = fx / df
        x_new = x - step
        
        # Damping: if the step is too large, shrink it
        # A common heuristic: if |step| > |x|, halve it?
        # Or check if |f(x_new)| < |f(x)|.
        
        # To avoid divergence, let's use a simple damping factor if the step is large.
        if abs(step) > 10 * max(1.0, abs(x)):
            x_new = x - 0.1 * step
        elif abs(step) > 1 * max(1.0, abs(x)):
            x_new = x - 0.5 * step
            
        if not np.isfinite(x_new):
            break
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not np.isfinite(fx_new):
            # If we hit a singularity or overflow, back off
            x = x - 0.5 * step
            continue
            
        # If the new value is worse, use bisection-like step towards the previous best
        if abs(fx_new) > abs(fx):
            # Move halfway back towards x
            x_new = x - 0.5 * step
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not np.isfinite(fx_new):
                break
                
        x = x_new
        fx = fx_new
        
    return x
# EVOLVE-BLOCK-END
