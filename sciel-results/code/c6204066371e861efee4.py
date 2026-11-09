"""
Root finder for 3 fixed oscillatory equations with call-budget awareness.
Strategy: Adaptive Newton-Bisection hybrid.
1. Check x0.
2. Use Newton's method with a bracketing safeguard.
   - If Newton step stays within a known bracket, take it.
   - If Newton step leaves the bracket or fails, bisect.
   - If no initial bracket is found, use a geometric expansion to find one.
3. This minimizes calls by leveraging the smoothness of oscillatory functions
   near roots (where Newton converges quadratically) while maintaining robustness
   via bisection.
"""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, starting near x0.
    Uses an adaptive Newton-Bisection hybrid to minimize function calls.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    tol = 1e-10  # Tighter than required 1e-8 to ensure pass
    
    x = float(x0)
    cache = {}
    call_count = 0
    
    def cached_f(x_val):
        nonlocal call_count
        x_key = float(x_val)
        if x_key in cache:
            return cache[x_key]
        try:
            result = f(x_key)
            call_count += 1
            if result is None or not math.isfinite(result):
                result = float('inf')
            cache[x_key] = result
            return result
        except Exception:
            call_count += 1
            cache[x_key] = float('inf')
            return float('inf')

    # Initial evaluation
    best_x = x
    best_f = cached_f(x)
    best_abs = abs(best_f) if math.isfinite(best_f) else float('inf')
    
    if best_abs <= tol:
        return best_x

    # If x0 is not a root, we need to find a bracket [a, b] such that f(a)*f(b) < 0
    # or find a point where f is very small and polish it.
    
    # Strategy:
    # 1. Try to establish a bracket around x0 using small steps.
    # 2. If a bracket is found, run the hybrid Newton-Bisection loop.
    # 3. If no bracket is found quickly, expand the search range geometrically.
    
    def find_bracket(center, f_center):
        """
        Try to find a bracket [a, b] containing a root near 'center'.
        Returns (a, b, f_a, f_b) or None.
        """
        if not math.isfinite(f_center):
            return None
            
        # Try small steps first
        step = 1e-4 * max(1.0, abs(center))
        for i in range(10):
            a = center - step
            b = center + step
            f_a = cached_f(a)
            f_b = cached_f(b)
            
            if math.isfinite(f_a) and math.isfinite(f_b):
                if f_a * f_b < 0:
                    return (a, b, f_a, f_b)
                # Check if a or b is close enough
                if abs(f_a) < best_abs:
                    best_x = a
                    best_abs = abs(f_a)
                if abs(f_b) < best_abs:
                    best_x = b
                    best_abs = abs(f_b)
                if best_abs <= tol:
                    return (a, b, f_a, f_b) # Signal success via best_x update, though logic below handles it
            
            step *= 2.0
            if time.time() > deadline - 10.0:
                break
        return None

    # Attempt to find an initial bracket
    bracket = find_bracket(x, best_f)
    
    if bracket is not None:
        a, b, f_a, f_b = bracket
        # If we found a point with |f| <= tol in the process, return it
        if best_abs <= tol:
            return best_x
    else:
        # If no local bracket, do a wider geometric search for a sign change
        # This is less efficient but necessary if x0 is far from a root or in a flat region
        scale = 1e-2 * max(1.0, abs(x))
        found = False
        for i in range(20):
            if time.time() > deadline - 5.0:
                break
            a = x - scale
            b = x + scale
            f_a = cached_f(a)
            f_b = cached_f(b)
            
            if math.isfinite(f_a) and math.isfinite(f_b):
                if abs(f_a) < best_abs:
                    best_x = a
                    best_abs = abs(f_a)
                if abs(f_b) < best_abs:
                    best_x = b
                    best_abs = abs(f_b)
                
                if best_abs <= tol:
                    return best_x
                    
                if f_a * f_b < 0:
                    bracket = (a, b, f_a, f_b)
                    found = True
                    break
            
            scale *= 2.0
        
        if not found:
            # No bracket found. Return best guess.
            # This is a failure case for the root finding, but we return the best x found.
            return best_x

    # Now we have a bracket [a, b] with f(a)*f(b) < 0
    a, b, f_a, f_b = bracket
    
    # Hybrid Newton-Bisection
    # We want to minimize calls. Newton is fast (quadratic), Bisection is slow (linear) but robust.
    # We will try Newton steps. If the Newton step stays inside [a,b], we take it.
    # Otherwise, we bisect.
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline - 2.0:
            break
            
        # Check if current a or b is good enough
        if abs(f_a) <= tol:
            return a
        if abs(f_b) <= tol:
            return b
            
        mid = 0.5 * (a + b)
        
        # Try Newton step from the point with smaller |f|
        if abs(f_a) < abs(f_b):
            x_curr = a
            f_curr = f_a
        else:
            x_curr = b
            f_curr = f_b
            
        # Estimate derivative using central difference if possible, or forward
        # To save calls, we can reuse f_a or f_b if they are close to x_curr?
        # No, we need f(x_curr + h).
        # Let's use a small h relative to the bracket size or absolute scale
        h = 1e-6 * max(1.0, abs(x_curr))
        
        # We might have f(x_curr + h) in cache? Unlikely unless h is tiny.
        f_plus = cached_f(x_curr + h)
        
        if not math.isfinite(f_plus):
            # Derivative estimation failed, bisect
            mid_val = cached_f(mid)
            if math.isfinite(mid_val):
                if abs(mid_val) <= tol:
                    return mid
                if f_a * mid_val <= 0:
                    b = mid
                    f_b = mid_val
                else:
                    a = mid
                    f_a = mid_val
            continue
            
        d = (f_plus - f_curr) / h
        
        if d == 0 or not math.isfinite(d):
            # Derivative is zero or invalid, bisect
            mid_val = cached_f(mid)
            if math.isfinite(mid_val):
                if abs(mid_val) <= tol:
                    return mid
                if f_a * mid_val <= 0:
                    b = mid
                    f_b = mid_val
                else:
                    a = mid
                    f_a = mid_val
            continue
            
        x_newton = x_curr - f_curr / d
        
        # Check if x_newton is within [a, b]
        if a < x_newton < b:
            # Take Newton step
            f_newton = cached_f(x_newton)
            
            if math.isfinite(f_newton):
                if abs(f_newton) <= tol:
                    return x_newton
                
                if f_a * f_newton <= 0:
                    b = x_newton
                    f_b = f_newton
                else:
                    a = x_newton
                    f_a = f_newton
            else:
                # Newton point invalid, bisect
                mid_val = cached_f(mid)
                if math.isfinite(mid_val):
                    if abs(mid_val) <= tol:
                        return mid
                    if f_a * mid_val <= 0:
                        b = mid
                        f_b = mid_val
                    else:
                        a = mid
                        f_a = mid_val
        else:
            # Newton step outside bracket, bisect
            mid_val = cached_f(mid)
            if math.isfinite(mid_val):
                if abs(mid_val) <= tol:
                    return mid
                if f_a * mid_val <= 0:
                    b = mid
                    f_b = mid_val
                else:
                    a = mid
                    f_a = mid_val
        
        # Update best
        if abs(f_a) < best_abs:
            best_abs = abs(f_a)
            best_x = a
        if abs(f_b) < best_abs:
            best_abs = abs(f_b)
            best_x = b
            
        if best_abs <= tol:
            return best_x

    # If loop finishes, return best found
    return best_x

# EVOLVE-BLOCK-END
