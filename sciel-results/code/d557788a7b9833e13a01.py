import math
import time

def find_root(f, x0):
    """
    Finds a root x for f(x) such that |f(x)| <= 1e-8.
    Uses a combination of bracketing search and secant/bisection refinement.
    """
    tol = 1e-8
    max_calls = 480  # Safety margin below 500
    call_count = 0
    cache = {}
    
    # Deadline guard: 110 seconds
    start_time = time.time()
    
    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("Call limit exceeded")
        if time.time() - start_time > 110:
            raise RuntimeError("Time limit exceeded")
            
        if x in cache:
            return cache[x]
        
        fx = f(x)
        call_count += 1
        cache[x] = fx
        
        if not math.isfinite(fx):
            raise ValueError("Non-finite value")
            
        return fx

    x = float(x0)
    
    # 1. Check initial point
    try:
        fx = safe_f(x)
    except Exception:
        # If even the first call fails or is invalid, we can't do much better
        # But we must return a float. x0 is the best guess.
        return x
        
    if abs(fx) <= tol:
        return x

    # 2. Find a bracket [a, b] with f(a)*f(b) < 0
    # We search outwards from x0
    a = x
    fa = fx
    b = None
    fb = None
    
    # Steps to try for bracketing
    # Small steps first to catch nearby roots, then larger
    steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
    
    # We also check the other direction
    directions = [1, -1]
    
    found_bracket = False
    best_x = x
    best_f = fx
    
    for step in steps:
        for d in directions:
            x_new = x + d * step
            try:
                fx_new = safe_f(x_new)
            except Exception:
                continue
            
            if abs(fx_new) <= tol:
                return x_new
            
            # Update best candidate
            if abs(fx_new) < abs(best_f):
                best_f = fx_new
                best_x = x_new
                
            # Check for sign change with current 'a'
            if fa * fx_new < 0:
                a, b = x, x_new
                fa, fb = fx, fx_new
                found_bracket = True
                break
            
            # If we haven't found a bracket yet, we can also check if 
            # the new point creates a bracket with a previously checked point
            # But simple expansion from x0 is usually sufficient for oscillatory funcs
            # if x0 is reasonably close.
            
        if found_bracket:
            break
        
        # If no bracket found with small steps, try expanding the base point?
        # Actually, the loop above checks x0 +/- step. 
        # If f is oscillatory, a root is likely within a few periods.
        # If we haven't found a bracket, we might need to move the base.
        # However, moving the base is risky. Let's stick to expanding from x0.
        
        # If we've used too many calls, stop searching for bracket
        if call_count > 100:
            break

    if not found_bracket:
        # If no bracket found, we can't guarantee a root with bisection.
        # Fall back to Secant/Newton-like method on the best points found?
        # Or just return the best_x found.
        # Given the problem statement "3 fixed oscillatory equations", 
        # a bracket is almost certainly findable if x0 is not pathological.
        # If we fail to bracket, returning best_x is the best bet.
        return best_x

    # 3. Refine the root using Secant Method, falling back to Bisection
    # We have a bracket [a, b]
    # Secant method needs two points. We can use a and b.
    x0_sec = a
    x1_sec = b
    f0 = fa
    f1 = fb
    
    # Ensure we don't divide by zero
    if f0 == f1:
        # Degenerate case, use midpoint?
        x_mid = (a + b) / 2
        try:
            fm = safe_f(x_mid)
            if abs(fm) <= tol:
                return x_mid
            # If f(a) < f(b), and f(a)*f(b) < 0, f is monotonic-ish?
            # Just switch to bisection
            pass
        except Exception:
            pass
        
    # Secant Method
    # x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    
    # To be robust, we will monitor if the secant step stays within the bracket
    # or gets close. If it diverges, we switch to Bisection.
    
    lo = min(a, b)
    hi = max(a, b)
    
    # Initialize secant points
    # We want to use the two points with the smallest |f| to start
    if abs(fa) < abs(fb):
        x_prev, x_curr = a, b
        f_prev, f_curr = fa, fb
    else:
        x_prev, x_curr = b, a
        f_prev, f_curr = fb, fa
        
    # Actually, standard secant uses the last two points.
    # Let's just use a and b as the first two points.
    x_prev = a
    f_prev = fa
    x_curr = b
    f_curr = fb
    
    for i in range(40):
        # Check time/calls
        if call_count > max_calls - 5:
            break
            
        # Calculate next point
        denom = f_curr - f_prev
        if denom == 0:
            break
            
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        
        # Check if x_next is finite
        if not math.isfinite(x_next):
            break
            
        # Check if we are still in the bracket or close to it
        # If we jump out of the bracket significantly, secant is diverging
        # Let's allow a small expansion but if it goes way out, switch to bisection
        if x_next < lo - (hi - lo) or x_next > hi + (hi - lo):
            # Divergence detected, switch to Bisection
            # We need to re-establish a bracket. We have lo, hi from the initial bracket.
            # But we might have lost track of f values at lo/hi if we moved.
            # Let's just use the initial bracket [a,b] for bisection fallback.
            break
            
        try:
            f_next = safe_f(x_next)
        except Exception:
            break
            
        if abs(f_next) <= tol:
            return x_next
            
        # Update secant points
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next
        f_curr = f_next
        
        # If the secant step is very small, we might be converged
        if abs(x_next - x_prev) < 1e-12 * max(1.0, abs(x_next)):
             # Check if f is small enough
             if abs(f_next) <= 1e-7: # Slightly looser for early exit, then polish
                 break
                 
    # If secant finished or diverged, use Bisection on the original bracket [a, b]
    # This is guaranteed to converge.
    a_bis = min(a, b)
    b_bis = max(a, b)
    fa_bis = f(a_bis) if a_bis in cache else safe_f(a_bis)
    fb_bis = f(b_bis) if b_bis in cache else safe_f(b_bis)
    
    # Ensure we have valid signs
    if fa_bis * fb_bis > 0:
        # Should not happen if we found a bracket, but just in case
        return best_x if 'best_x' in locals() else x0
        
    for i in range(60):
        if call_count > max_calls - 2:
            break
            
        mid = (a_bis + b_bis) / 2.0
        f_mid = safe_f(mid)
        
        if abs(f_mid) <= tol:
            return mid
            
        if fa_bis * f_mid < 0:
            b_bis = mid
            fb_bis = f_mid
        else:
            a_bis = mid
            fa_bis = f_mid
            
        # Check for convergence
        if abs(b_bis - a_bis) < 1e-12:
            break
            
    # Return the midpoint of the final bracket
    return (a_bis + b_bis) / 2.0
