"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy: Adaptive Secant with Multi-Scale Seeding and Newton Polish.
    We leverage the oscillatory nature of the functions by testing multiple
    initial secant pairs at different scales. We also use a Newton-based
    polish at the end to ensure high precision if the secant method gets
    close but doesn't quite meet the strict tolerance.
    """
    deadline = time.time() + 145.0
    tol = 1e-9  # Target |f(x)| <= 1e-8, use tighter internal tol
    x0 = float(x0)
    
    # Global best tracking
    best_x = x0
    try:
        best_f = f(x0)
    except Exception:
        best_f = float('inf')
    best_abs = abs(best_f) if math.isfinite(best_f) else float('inf')
    
    if best_abs <= tol:
        return best_x

    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if x_val is None or f_val is None:
            return
        if not math.isfinite(x_val) or not math.isfinite(f_val):
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
    # We prioritize medium scales first as they often catch oscillations better than tiny or huge ones.
    # But we keep a wide range.
    scales = [1e-2, 1e-1, 1.0, 1e-3, 1e-4, 10.0, 1e-5, 100.0, 1000.0, 10000.0]
    
    # Define a list of trials (scale, sign_a, sign_b)
    # We want to explore different initial brackets.
    trials = []
    for s in scales:
        # Trial 1: (x0+s, x0+2s) - forward stepping
        trials.append((s, 1, 2))
        # Trial 2: (x0-s, x0+s) - symmetric bracket
        trials.append((s, -1, 1))
        # Trial 3: (x0, x0+s) - using x0 itself as one point
        trials.append((s, 0, 1))
        
    # Run secant method for each trial
    # We limit iterations per trial to keep total calls down, but allow enough for convergence
    # Secant converges superlinearly, so 30-40 iters is plenty unless it diverges.
    
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
        x_curr_a = x_a
        x_curr_b = x_b
        f_curr_a = f_a
        f_curr_b = f_b
        
        for _ in range(40):
            if time.time() > deadline:
                break
                
            # Check convergence of current points
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
                
            # Shift window
            x_curr_a = x_curr_b
            f_curr_a = f_curr_b
            x_curr_b = x_new
            f_curr_b = f(x_new)
            
            update_best(x_curr_b, f_curr_b)
            
            if best_abs <= tol:
                return best_x

    # If we haven't found a root with |f| <= tol, we try a Newton polish
    # on the best point found so far. This can help if the secant method
    # converged to a point with small f(x) but not small enough, or if
    # the derivative information can be leveraged for a final kick.
    
    x = best_x
    for _ in range(20):
        if time.time() > deadline:
            break
        try:
            fx = f(x)
        except Exception:
            break
        update_best(x, fx)
        if abs(fx) <= tol:
            return x
        
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx2 = f(x + h)
        except Exception:
            break
        update_best(x + h, fx2)
        
        d = (fx2 - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        # Newton step
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
        
        # Damping: if step is too large, shrink it
        max_step = 10.0 * max(1.0, abs(x))
        step = x_new - x
        if abs(step) > max_step:
            x_new = x + math.copysign(max_step, step)
        
        x = x_new

    return best_x
# EVOLVE-BLOCK-END
