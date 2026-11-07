import math
import time
from scipy.optimize import brentq, minimize_scalar

# Global state for the specific run, though we can also pass it via closure
# Since find_root is called 3 times, we need to be careful with state if we want to share info, 
# but the problem says "3 fixed equations that are hidden", implying f is different each time.
# So we treat each call independently.

def find_root(f, x0):
    """
    Finds a root x such that |f(x)| <= 1e-8.
    Uses a bracketing strategy followed by Brent's method.
    """
    start_time = time.time()
    MAX_TIME = 110.0  # Safety margin before 120s limit
    MAX_CALLS = 450   # Safety margin before 500 limit
    
    call_count = 0
    best_x = float(x0)
    best_fx = None
    best_abs_fx = float('inf')

    def count_f(x):
        nonlocal call_count, best_x, best_fx, best_abs_fx
        if time.time() - start_time > MAX_TIME:
            raise TimeoutError
        if call_count >= MAX_CALLS:
            raise RuntimeError("Call limit exceeded")
        
        fx = f(float(x))
        call_count += 1
        
        if not math.isfinite(fx):
            return fx
            
        abs_fx = abs(fx)
        if abs_fx < best_abs_fx:
            best_abs_fx = abs_fx
            best_x = float(x)
            best_fx = fx
            
        return fx

    # 1. Check initial point
    try:
        fx0 = count_f(x0)
    except (TimeoutError, RuntimeError):
        return best_x

    if abs(fx0) <= 1e-8:
        return best_x

    # 2. Find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = None
    b = None
    fa = None
    fb = None
    
    # We will expand a window around x0
    # Start with a small step
    h = 1e-4
    max_h = 1e6
    
    # Directions: + and -
    # We check points x0 + h, x0 - h, then expand h
    
    # To be efficient, let's check a sequence of points
    # x0, x0+h, x0-h, x0+10h, x0-10h, ...
    
    current_h = 1e-4
    found_bracket = False
    
    while current_h <= max_h:
        if time.time() - start_time > MAX_TIME:
            break
            
        # Check right side
        x_r = x0 + current_h
        try:
            fx_r = count_f(x_r)
        except (TimeoutError, RuntimeError):
            break
            
        if math.isfinite(fx_r):
            if abs(fx_r) <= 1e-8:
                return x_r
            if fx0 * fx_r < 0:
                a, b = x0, x_r
                fa, fb = fx0, fx_r
                found_bracket = True
                break
                
        # Check left side
        x_l = x0 - current_h
        try:
            fx_l = count_f(x_l)
        except (TimeoutError, RuntimeError):
            break
            
        if math.isfinite(fx_l):
            if abs(fx_l) <= 1e-8:
                return x_l
            if fx0 * fx_l < 0:
                a, b = x_l, x0
                fa, fb = fx_l, fx0
                found_bracket = True
                break
            
        # If no sign change with x0, check if sign change between x_l and x_r
        # This handles cases where x0 is not between the roots but the interval is
        if math.isfinite(fx_l) and math.isfinite(fx_r):
            if fx_l * fx_r < 0:
                a, b = x_l, x_r
                fa, fb = fx_l, fx_r
                found_bracket = True
                break

        current_h *= 10.0

    # 3. If bracket found, use Brent's method
    if found_bracket:
        try:
            # brentq requires f(a) and f(b) to have opposite signs
            # We already checked this, but brentq will check again.
            # xtol and rtol control precision.
            root = brentq(count_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root):
                return root
        except (ValueError, RuntimeError, TimeoutError):
            # If brentq fails, fall through to best_x
            pass

    # 4. Fallback: Minimize |f(x)| in a local region
    # This is less efficient but might find a root if sign change was missed (e.g. touching zero)
    # or if the function is non-monotonic in the bracket.
    # However, for oscillatory functions, sign change is the standard.
    # Let's try a bounded minimization around x0.
    
    # Determine bounds for minimization based on last successful h
    # If we expanded to h, search in [x0-h, x0+h]
    search_h = max(current_h, 1e-2)
    lower = x0 - search_h
    upper = x0 + search_h
    
    def obj(x):
        if time.time() - start_time > MAX_TIME:
            raise TimeoutError
        if call_count >= MAX_CALLS:
            raise RuntimeError("Call limit exceeded")
        fx = f(float(x))
        call_count += 1
        if not math.isfinite(fx):
            return 1e100
        abs_fx = abs(fx)
        if abs_fx < best_abs_fx:
            best_abs_fx = abs_fx
            best_x = float(x)
        return abs_fx

    try:
        res = minimize_scalar(obj, bounds=(lower, upper), method='bounded', options={'xatol': 1e-10})
        if res.success and math.isfinite(res.x):
            # Verify the result
            fx_res = count_f(res.x)
            if abs(fx_res) <= 1e-8:
                return res.x
            # Even if not <= 1e-8, if it's better than best_x, return it?
            # The task requires |f(x)| <= 1e-8 for success.
            # If minimize_scalar finds a point with |f| > 1e-8, it's a failure unless it's the best we have.
            # But we need to return a float. Returning a point with |f| > 1e-8 is a failure.
            # So we only return res.x if it's a valid root.
            # Otherwise, we return best_x which might also be invalid.
            # The evaluator checks |f(x)| <= 1e-8.
            if abs(fx_res) <= 1e-8:
                 return res.x
    except (TimeoutError, RuntimeError, ValueError):
        pass

    # 5. Return the best point found so far
    # If best_abs_fx > 1e-8, this will be a failure, but it's the best we can do.
    return best_x
