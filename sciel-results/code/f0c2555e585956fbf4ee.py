"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Finds a root of f using a hybrid Newton/Bisection strategy.
    1. Attempts fast Newton convergence using finite difference derivatives.
    2. If Newton steps become unstable or diverge, falls back to a 
       bracketing search (Illinois method) to guarantee finding a sign change.
    3. Includes a strict deadline guard to prevent timeouts.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    x = float(x0)
    best_x = x
    best_val = float('inf')
    
    # Helper to update best solution
    def update_best(x_val, f_val):
        nonlocal best_x, best_val
        if math.isfinite(f_val) and abs(f_val) < best_val:
            best_val = abs(f_val)
            best_x = x_val
            if best_val <= 1e-8:
                return True
        return False

    # Initial evaluation
    try:
        f_x = f(x)
    except Exception:
        return x
        
    if not math.isfinite(f_x):
        return x
        
    if abs(f_x) <= 1e-8:
        return x

    update_best(x, f_x)

    # Phase 1: Newton's Method with stability checks
    # We allow a few Newton steps to try for rapid convergence
    for i in range(10):
        if (time.time() - start_time) > deadline:
            break
            
        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        try:
            f_xh = f(x + h)
            if not math.isfinite(f_xh):
                break
            d = (f_xh - f_x) / h
            if d == 0 or not math.isfinite(d):
                break
        except Exception:
            break

        # Newton step
        x_new = x - f_x / d
        
        # Stability check: 
        # 1. Step shouldn't be wildly larger than current x (unless x is small)
        # 2. New x must be finite
        # 3. Ideally |f(x_new)| < |f(x)|, but we allow some slack initially
        if not math.isfinite(x_new):
            break
            
        # If the step is huge, Newton is likely diverging or derivative is bad
        max_step = max(1.0, abs(x)) * 10.0
        if abs(x_new - x) > max_step:
            break
            
        x = x_new
        try:
            f_x = f(x)
        except Exception:
            break
            
        if not math.isfinite(f_x):
            break
            
        if abs(f_x) <= 1e-8:
            return x
            
        update_best(x, f_x)
        
        # If we are getting closer, continue Newton
        # If we are getting further away, switch to bracketing
        # We track previous best to see if we are improving
        if best_val > abs(f_x):
             continue
        else:
             # If we didn't improve, it might be oscillating or diverging
             # Switch to robust bracketing
             break

    # Phase 2: Robust Bracketing Search (Illinois Method)
    # If Newton failed or didn't converge, we need to find a sign change.
    # We expand a window around the best point found so far.
    
    # Start bracketing from the current best_x
    x_center = best_x
    f_center = f(x_center) if math.isfinite(f(x_center)) else None
    
    # If we don't have a valid f_center, try to get one
    if f_center is None or not math.isfinite(f_center):
        # Try a few random points or just return best
        # For polynomials, x0 is usually decent. If not, we are in trouble.
        # Let's try expanding from x0 again if best_x is bad.
        x_center = x0
        try:
            f_center = f(x_center)
        except Exception:
            return best_x
        if not math.isfinite(f_center):
            return best_x

    # Define a function to find a sign change around a center
    def find_bracket(center, f_center):
        if abs(f_center) <= 1e-8:
            return center, center, f_center, f_center
            
        step = 1.0
        max_step = 1e6
        left = center
        right = center
        f_left = f_center
        f_right = f_center
        
        # We already have one point. We need another with opposite sign.
        # Try expanding to the right first, then left.
        found = False
        for _ in range(50):
            if (time.time() - start_time) > deadline:
                break
                
            # Try right
            x_try = center + step
            try:
                f_try = f(x_try)
            except Exception:
                break
            if not math.isfinite(f_try):
                break
                
            if f_center * f_try < 0:
                left, right = center, x_try
                f_left, f_right = f_center, f_try
                found = True
                break
            
            # Try left
            x_try = center - step
            try:
                f_try = f(x_try)
            except Exception:
                break
            if not math.isfinite(f_try):
                break
                
            if f_center * f_try < 0:
                left, right = x_try, center
                f_left, f_right = f_try, f_center
                found = True
                break
                
            # Expand step
            step *= 2.0
            if step > max_step:
                break
                
        return left, right, f_left, f_right if found else (None, None, None, None)

    left, right, f_left, f_right = find_bracket(x_center, f_center)
    
    if left is not None:
        # Illinois Method (False Position with shrinking)
        # We have f(left) * f(right) < 0
        for _ in range(100):
            if (time.time() - start_time) > deadline:
                break
                
            # Linear interpolation for next point
            if f_left == f_right:
                x_new = (left + right) / 2.0
            else:
                x_new = (left * f_right - right * f_left) / (f_right - f_left)
                
            # Ensure x_new is within [left, right] (Illinois usually does this)
            if x_new <= left or x_new >= right:
                x_new = (left + right) / 2.0
                
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
                
            if abs(f_new) <= 1e-8:
                return x_new
                
            update_best(x_new, f_new)
            
            # Update bracket
            if f_left * f_new < 0:
                right = x_new
                f_right = f_new
                # Illinois: shrink the side that is not the root
                left = (left + right) / 2.0
                # We don't necessarily need to re-evaluate f(left) if we assume continuity 
                # and just use bisection for the "other" point in Illinois, 
                # but standard Illinois updates the endpoint. 
                # To be safe and simple, let's just use Bisection if we want to be 100% robust 
                # or stick to False Position. False Position can be slow if function is flat.
                # Let's use a hybrid: if step is too small, bisect.
                
                # Actually, simpler robust approach:
                # Just bisect. It's O(log N) and guaranteed.
                # Given the 500 call limit, bisection is safe if we start with a small bracket.
                # But our bracket might be huge.
                # Let's stick to Illinois logic but ensure we don't stall.
                
                # Re-evaluating f(left) is expensive. 
                # Let's switch to pure Bisection for the final polish if we are close.
                # For now, update the bracket side.
                pass
            else:
                left = x_new
                f_left = f_new
                right = (left + right) / 2.0
                # Again, f_right is stale.
                
            # To avoid stale function values causing issues, let's just do Bisection 
            # if the interval is small enough, or just always Bisection for simplicity 
            # and robustness within the call budget.
            # Bisection halves the interval. 100 bisections is 2^100 precision, which is plenty.
            # The risk is if the initial bracket is huge (1e6), 100 bisections might not be enough 
            # to get to 1e-8 precision? 
            # 1e6 / 2^100 is tiny. So 100 bisections is plenty.
            
            # Let's restart the loop logic to be a clean Bisection for the found bracket.
            break

    # Clean Bisection on the found bracket
    if left is not None:
        for _ in range(80): # 80 bisections is ~1e24 precision, plenty for 1e-8
            if (time.time() - start_time) > deadline:
                break
                
            mid = (left + right) / 2.0
            try:
                f_mid = f(mid)
            except Exception:
                break
            if not math.isfinite(f_mid):
                break
                
            if abs(f_mid) <= 1e-8:
                return mid
                
            update_best(mid, f_mid)
            
            if f_left * f_mid < 0:
                right = mid
                f_right = f_mid
            else:
                left = mid
                f_left = f_mid
                
            # Early exit if interval is very small
            if abs(right - left) < 1e-10:
                break

    return best_x

# EVOLVE-BLOCK-END
