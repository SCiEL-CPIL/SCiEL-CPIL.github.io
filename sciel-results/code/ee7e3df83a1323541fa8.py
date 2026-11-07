import time
import math
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    
    # Helper to check time and return best x if exceeded
    def check_time(best_x, best_f):
        if time.time() - start_time > TIME_LIMIT:
            return best_x
        return None

    # Phase 1: Local Secant Method
    # This exploits the proximity of x0 to the root.
    
    # Get initial points
    h = 1e-6 * max(1.0, abs(x0))
    x1 = x0 + h
    x2 = x0 - h  # Use symmetric points for better derivative estimate if needed, 
                 # but Secant uses two points. Let's stick to standard Secant.
    
    # Actually, let's just use x0 and x0+h.
    x_prev = x0
    f_prev = f(x_prev)
    if abs(f_prev) <= 1e-8:
        return x_prev

    x_curr = x0 + h
    f_curr = f(x_curr)
    if abs(f_curr) <= 1e-8:
        return x_curr

    best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
    best_f = min(abs(f_prev), abs(f_curr))

    # Secant iterations
    for i in range(20):
        # Time check
        res = check_time(best_x, best_f)
        if res is not None:
            return res

        if f_curr == f_prev:
            break  # Derivative zero or flat, Secant fails
            
        # Secant step
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # Check for non-finite or huge jumps
        if not math.isfinite(x_new):
            break
            
        # If the step is huge, it might be diverging. 
        # Heuristic: if |x_new - x_curr| > 10 * max(1, |x_curr|), consider it diverging
        if abs(x_new - x_curr) > 10.0 * max(1.0, abs(x_curr)):
             break

        f_new = f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new

        # Update best
        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new

        # Check for stagnation
        if abs(x_new - x_curr) < 1e-12 * max(1.0, abs(x_curr)):
            break

        # Update points
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new

    # Phase 2: Robust Bracketing and Brent's Method
    # If Secant didn't find a root, we need to find a bracket [a, b] where f(a)*f(b) < 0
    # We start from the best_x found so far.
    
    a = best_x
    fa = f(a) # We might have already calculated this, but to be safe and simple, 
              # we can reuse if we tracked it. Let's just recalc or track.
              # Actually, we can track fa, fb in the loop above? 
              # It's easier to just restart bracketing from best_x.
    
    # We need f(a). We have best_x, but do we have f(best_x)?
    # We tracked best_f, but not the sign. Let's just call f(best_x) again? 
    # That wastes 1 call. 
    # Better: In the Secant loop, we always know f_prev, f_curr, f_new.
    # If best_x was x_new, we have f_new.
    # To simplify code and avoid state bugs, I will just call f(best_x) here.
    # Cost: 1 call. Acceptable.
    
    fa = f(best_x)
    if abs(fa) <= 1e-8:
        return best_x

    # Expand interval to find a sign change
    # Start with a small step
    step = 1e-3 * max(1.0, abs(best_x))
    found_bracket = False
    left = best_x
    right = best_x
    f_left = fa
    f_right = fa
    
    # We need to explore both directions if the root is on either side, 
    # but usually one direction works. Let's try expanding outwards.
    
    # Try positive direction first
    for i in range(30):
        res = check_time(best_x, best_f)
        if res is not None:
            return res
            
        right = best_x + step
        f_right = f(right)
        
        if abs(f_right) <= 1e-8:
            return right
            
        if fa * f_right < 0:
            # Found bracket [best_x, right]
            a = best_x
            b = right
            found_bracket = True
            break
            
        # If we haven't found a sign change, keep expanding
        # But also check if we are going away from a root? 
        # For oscillatory, we just need *a* root.
        
        step *= 2.0
        if step > 1e6: # Cap expansion
            break

    if not found_bracket:
        # Try negative direction
        step = 1e-3 * max(1.0, abs(best_x))
        for i in range(30):
            res = check_time(best_x, best_f)
            if res is not None:
                return res
                
            left = best_x - step
            f_left = f(left)
            
            if abs(f_left) <= 1e-8:
                return left
                
            if fa * f_left < 0:
                # Found bracket [left, best_x]
                a = left
                b = best_x
                found_bracket = True
                break
                
            step *= 2.0
            if step > 1e6:
                break

    if found_bracket:
        # Use Brent's method
        try:
            # maxiter is high to ensure convergence, but brentq is fast
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(f(root)) <= 1e-8:
                return root
        except Exception:
            pass

    # Fallback: Return the best x found so far
    return best_x
