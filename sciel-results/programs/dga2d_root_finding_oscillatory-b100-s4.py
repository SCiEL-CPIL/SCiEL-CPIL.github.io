"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy: Global Secant Search.
    We test multiple initial secant pairs at different scales to find a 
    convergent path quickly, leveraging the oscillatory nature of the functions.
    """
    deadline = time.time() + 145.0
    tol = 1e-9
    x0 = float(x0)
    
    # Global best tracking
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f) if math.isfinite(best_f) else float('inf')
    
    if best_abs <= tol:
        return best_x

    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if not math.isfinite(x_val) or f_val is None or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f

    # We will try a set of "seeds" for secant method.
    # Each seed is a pair (x_a, x_b).
    # We generate pairs based on offsets from x0.
    # Offsets are logarithmically spaced to cover local and global scales.
    
    # Scale factors: from very local (1e-5) to very global (1e4)
    # We use a geometric progression.
    scales = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    
    # For each scale, we try a few relative offsets to avoid symmetry issues
    # e.g., if f is even, x0+s and x0-s might be symmetric.
    # We use pairs like (x0 + s, x0 + 2s) or (x0 - s, x0 + s).
    
    # Let's define a list of (scale, sign1, sign2, ratio)
    # We want to explore different initial brackets.
    trials = []
    for s in scales:
        # Trial 1: (x0+s, x0+2s) - forward stepping
        trials.append((s, 1, 2))
        # Trial 2: (x0-s, x0+s) - symmetric bracket (if s is small enough to catch oscillation)
        trials.append((s, -1, 1))
        # Trial 3: (x0+s, x0-s) - same as above but reversed
        # Trial 4: (x0, x0+s) - using x0 itself as one point
        trials.append((s, 0, 1))
        
    # Sort trials by scale, starting with smaller scales (local) then larger (global)
    # Actually, for oscillatory functions, a medium scale might be better than tiny.
    # But let's just iterate through them. We have a deadline, so we can stop early.
    
    # To maximize chance, we might interleave or just run them.
    # Running them sequentially is safer for state management.
    
    for scale, sign_a, sign_b in trials:
        if time.time() > deadline:
            break
            
        # Define the two starting points
        x_a = x0 + scale * sign_a
        x_b = x0 + scale * sign_b
        
        # If points are too close, skip to avoid division by zero or instability
        if abs(x_a - x_b) < 1e-12 * max(1.0, abs(x0)):
            continue
            
        # Evaluate f at starting points
        try:
            f_a = f(x_a)
            f_b = f(x_b)
        except Exception:
            continue
            
        update_best(x_a, f_a)
        update_best(x_b, f_b)
        
        if best_abs <= tol:
            return best_x
            
        # Check for immediate success
        if abs(f_a) <= tol:
            return x_a
        if abs(f_b) <= tol:
            return x_b
            
        # Secant method iteration
        # We limit iterations to keep total calls down, but allow enough for convergence
        # Secant converges superlinearly (~1.6), so 30-40 iters is plenty for double precision
        # unless it diverges.
        
        x_curr_a = x_a
        x_curr_b = x_b
        f_curr_a = f_a
        f_curr_b = f_b
        
        for _ in range(40):
            if time.time() > deadline:
                break
                
            # Check convergence of current points (already done in update_best, but check explicit tol)
            # The update_best handles the global best. We just need to see if THIS path found a root.
            if abs(f_curr_a) <= tol:
                return x_curr_a
            if abs(f_curr_b) <= tol:
                return x_curr_b
                
            # Calculate next point
            # x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            denom = f_curr_b - f_curr_a
            if denom == 0 or not math.isfinite(denom):
                break
                
            x_new = x_curr_b - f_curr_b * (x_curr_b - x_curr_a) / denom
            
            if not math.isfinite(x_new):
                break
                
            # Optional: Damping if step is huge
            # If the step is extremely large, it might be a sign of divergence.
            # But for oscillatory functions, large jumps can be good.
            # We'll just proceed.
            
            # Shift window
            x_curr_a = x_curr_b
            f_curr_a = f_curr_b
            x_curr_b = x_new
            f_curr_b = f(x_new)
            
            update_best(x_curr_b, f_curr_b)
            
            if best_abs <= tol:
                return best_x

    # If all trials failed to find |f| <= tol, return the best we found
    return best_x
# EVOLVE-BLOCK-END
