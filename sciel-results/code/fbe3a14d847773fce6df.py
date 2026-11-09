"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    eval_count = 0
    max_evals = 500
    
    def safe_f(x):
        nonlocal eval_count
        if time.time() > deadline:
            raise TimeoutError()
        if eval_count >= max_evals:
            raise ValueError("Max evals exceeded")
        eval_count += 1
        try:
            val = f(x)
            # Handle non-finite values by returning a large number with sign of last finite? 
            # Or just let it propagate? Usually ill-conditioned means steep, not NaN.
            # If NaN, it breaks comparisons. Let's assume f returns floats.
            return float(val)
        except Exception:
            # If f fails, return a large value to push bisection away?
            # Or raise? Raising crashes the program. 
            # Let's return a large number to treat as "not a root".
            return 1e308

    x = float(x0)
    
    # 1. Check initial point
    fx = safe_f(x)
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x

    # 2. Find a bracket [a, b] such that f(a) * f(b) < 0
    # Start with a small step and expand
    a = x
    fa = fx
    h = 1e-4
    b = None
    fb = None
    
    # Try to find a sign change by expanding interval
    # We evaluate f(x + h) and f(x - h)
    # To save evals, we can expand one side at a time or both.
    # Let's expand right first, then left.
    
    # If fa is very small but not small enough, we might be close.
    # But we need a bracket for robustness.
    
    # Strategy:
    # Keep a record of the best point (smallest |f|) found so far.
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    # Expand interval to find sign change
    # We use a geometric expansion: h, 2h, 4h...
    # We check x+h and x-h
    
    # To minimize evals, let's try a few steps.
    # If we find a sign change between current a and new point, we have a bracket.
    
    # Let's try a symmetric expansion or sequential.
    # Sequential is often better if the root is close.
    
    # Try right side
    for i in range(20):
        if eval_count >= max_evals or time.time() > deadline:
            break
        new_x = x + h
        new_fx = safe_f(new_x)
        if not math.isfinite(new_fx):
            h *= 2
            continue
            
        if abs(new_fx) < best_abs:
            best_x = new_x
            best_fx = new_fx
            best_abs = abs(new_fx)
            
        if fa * new_fx < 0:
            b = new_x
            fb = new_fx
            break
        elif abs(new_fx) <= 1e-8:
            return new_x
            
        # If no sign change, expand
        h *= 2
        if h > 1e10:
            break
            
    # If no bracket found on right, try left
    if b is None:
        h = 1e-4
        for i in range(20):
            if eval_count >= max_evals or time.time() > deadline:
                break
            new_x = x - h
            new_fx = safe_f(new_x)
            if not math.isfinite(new_fx):
                h *= 2
                continue
                
            if abs(new_fx) < best_abs:
                best_x = new_x
                best_fx = new_fx
                best_abs = abs(new_fx)
                
            if fa * new_fx < 0:
                b = new_x
                fb = new_fx
                break
            elif abs(new_fx) <= 1e-8:
                return new_x
                
            h *= 2
            if h > 1e10:
                break

    # If we found a bracket [a, b] with sign change
    if b is not None and math.isfinite(fb):
        # Ensure a is the left end
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Hybrid Newton-Bisection
        # Iterate until |f(x)| <= 1e-8 or max evals/time
        x = (a + b) / 2.0
        # We already have fa, fb. We need fx.
        # Let's start with the midpoint or use Newton from x0 if it's inside?
        # Newton from x0 is faster if x0 is inside bracket.
        if a < x0 < b:
            x = x0
            fx = fa
        else:
            x = (a + b) / 2.0
            fx = safe_f(x)
            
        for _ in range(100): # Max 100 iterations, each ~2 evals max
            if time.time() > deadline or eval_count >= max_evals:
                break
                
            if abs(fx) <= 1e-8:
                return x
                
            # Update best
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
                
            # Determine next step
            # Try Newton
            # Derivative estimate
            h = 1e-6 * max(1.0, abs(x))
            # Use central difference if possible, else forward
            # To save evals, forward difference is 1 eval. Central is 2.
            # Let's use forward difference to save budget, it's usually enough for ill-conditioned if we have bisection backup.
            f_xh = safe_f(x + h)
            if not math.isfinite(f_xh):
                # Fallback to bisection
                pass
            else:
                d = (f_xh - fx) / h
                
            if math.isfinite(d) and d != 0:
                x_new = x - fx / d
                # Check if x_new is inside bracket
                if a < x_new < b:
                    x = x_new
                    fx = safe_f(x)
                    continue
                else:
                    # Newton step outside, use bisection
                    pass
            
            # Bisection step
            if fa * fx < 0:
                b = x
                fb = fx
            else:
                a = x
                fa = fx
                
            x = (a + b) / 2.0
            fx = safe_f(x)
            
        # After loop, return best
        if abs(fx) <= 1e-8:
            return x
        return best_x

    # If no bracket found
    # We rely on the best point found during expansion
    # Or we can try a simple minimization of |f|
    # But given the task "root finding", a root likely exists.
    # If no bracket, maybe the function doesn't cross zero, or we missed it.
    # Return the point with smallest |f| found.
    return best_x

# EVOLVE-BLOCK-END
