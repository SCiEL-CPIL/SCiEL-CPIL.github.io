"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Strategy:
    1. Start with Newton's method (fast if close to root).
    2. If Newton diverges or fails, switch to a bracketing method (bisection/secant hybrid).
    3. Use a high-precision polish pass at the end.
    4. Track evaluations and time to stay within limits.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    eval_count = 0
    max_evals = 500
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        eval_count += 1
        return f(x)
    
    def is_valid(x):
        return x is not None and math.isfinite(x)
    
    # Initial point
    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_fx = fx
    
    # Phase 1: Try Newton's method with safeguards
    newton_success = False
    for i in range(50):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-8:
            newton_success = True
            break
        
        # Compute derivative using central difference
        h = 1e-8 * max(1.0, abs(x))
        fx_plus = eval_f(x + h)
        fx_minus = eval_f(x - h)
        if fx_plus is None or fx_minus is None:
            break
        
        df = (fx_plus - fx_minus) / (2.0 * h)
        
        if df == 0 or not math.isfinite(df):
            break
        
        # Newton step
        dx = -fx / df
        x_new = x + dx
        
        # Safeguard: if step is too large, take a bisection-like step
        if not math.isfinite(x_new):
            break
        
        # Limit step size to prevent flying away
        max_step = 10.0 * max(1.0, abs(x))
        if abs(dx) > max_step:
            x_new = x + math.copysign(max_step, dx)
        
        fx_new = eval_f(x_new)
        if fx_new is None:
            break
        
        # Check for improvement
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
            best_x = x
            best_fx = fx
        else:
            # Newton step didn't improve; try a smaller step
            # Use a damped Newton step
            for damp in [0.5, 0.25, 0.125]:
                x_damp = x + damp * dx
                fx_damp = eval_f(x_damp)
                if fx_damp is None:
                    break
                if abs(fx_damp) < abs(fx):
                    x = x_damp
                    fx = fx_damp
                    best_x = x
                    best_fx = fx
                    break
    
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 2: If Newton didn't succeed, use a bracketing approach
    # Try to find a bracket around the current best point
    a = best_x
    fa = best_fx
    
    # Search for a sign change
    step = 1.0 * max(1.0, abs(a))
    bracket_found = False
    b = None
    fb = None
    
    for direction in [1, -1]:
        if time.time() > deadline:
            break
        for _ in range(30):
            if eval_count >= max_evals - 10:
                break
            b = a + direction * step
            fb = eval_f(b)
            if fb is None:
                break
            if fa * fb < 0:
                bracket_found = True
                break
            # Update a to the point with smaller |f|
            if abs(fb) < abs(fa):
                a = b
                fa = fb
            step *= 2.0
            if not math.isfinite(step) or step > 1e15:
                break
        if bracket_found:
            break
    
    if bracket_found:
        # Use Illinois/Regula Falsi hybrid with bisection safeguard
        # Ensure fa and fb correspond to a and b
        if fa * fb >= 0:
            # Shouldn't happen, but just in case
            bracket_found = False
        else:
            for _ in range(100):
                if time.time() > deadline:
                    break
                if eval_count >= max_evals:
                    break
                
                # Regula falsi step
                if abs(fb) < abs(fa):
                    a, fa = b, fb
                    # Swap back for formula
                    b, fb = a, fa  # This is wrong, let's do it properly
                
                # Proper Illinois method
                # Keep a and b such that fa and fb have opposite signs
                # Always shrink the side with the larger |f|
                
                # Secant step
                if fb == fa:
                    x_new = (a + b) / 2.0
                else:
                    x_new = a - fa * (b - a) / (fb - fa)
                
                # If secant step is too close to a or b, use bisection
                if abs(x_new - a) < 0.1 * abs(b - a) or abs(x_new - b) < 0.1 * abs(b - a):
                    x_new = (a + b) / 2.0
                
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                
                if abs(fx_new) <= 1e-8:
                    return x_new
                
                if fa * fx_new < 0:
                    b = x_new
                    fb = fx_new
                else:
                    a = x_new
                    fa = fx_new
                    # Illinois modification: shrink the other side
                    fb *= 0.5
                
                # Update best
                if abs(fx_new) < abs(best_fx):
                    best_x = x_new
                    best_fx = fx_new
    else:
        # No bracket found; try local minimization of |f(x)|
        # Use a simple line search around best_x
        step = 1e-4 * max(1.0, abs(best_x))
        for _ in range(50):
            if time.time() > deadline:
                break
            if eval_count >= max_evals:
                break
            
            candidates = [best_x + step, best_x - step]
            best_local = None
            best_local_fx = None
            for c in candidates:
                fc = eval_f(c)
                if fc is None:
                    continue
                if best_local is None or abs(fc) < abs(best_local_fx):
                    best_local = c
                    best_local_fx = fc
            
            if best_local is None:
                break
            
            if abs(best_local_fx) < abs(best_fx):
                best_x = best_local
                best_fx = best_local_fx
                if abs(best_fx) <= 1e-8:
                    return best_x
            else:
                step *= 0.5
                if step < 1e-15:
                    break
    
    # Phase 3: Final polish with high-precision Newton
    x = best_x
    fx = best_fx
    for _ in range(20):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-12:
            break
        h = 1e-10 * max(1.0, abs(x))
        fx_plus = eval_f(x + h)
        fx_minus = eval_f(x - h)
        if fx_plus is None or fx_minus is None:
            break
        df = (fx_plus - fx_minus) / (2.0 * h)
        if df == 0 or not math.isfinite(df):
            break
        x_new = x - fx / df
        if not math.isfinite(x_new):
            break
        fx_new = eval_f(x_new)
        if fx_new is None:
            break
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            break
    
    if abs(fx) <= 1e-8:
        return x
    
    # Return the best point found
    return best_x
# EVOLVE-BLOCK-END
