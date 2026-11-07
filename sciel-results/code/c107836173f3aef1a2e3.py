"""Scalar root finder for 3 fixed smooth polynomial equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a multi-strategy approach with deadline guard.
    """
    DEADLINE = 140.0  # Safety margin before 150s kill
    start_time = time.time()
    
    def check_deadline():
        return (time.time() - start_time) < DEADLINE
    
    def is_finite(x):
        return math.isfinite(x)
    
    def safe_f(x):
        try:
            val = f(x)
            if not is_finite(val):
                return None
            return float(val)
        except:
            return None
    
    # Track best solution found
    best_x = None
    best_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if is_finite(x) and is_finite(fx) and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        if best_x is not None and abs(best_fx) <= 1e-8:
            return True
        return False
    
    # First try at x0
    x0 = float(x0)
    fx0 = safe_f(x0)
    if fx0 is not None:
        update_best(x0, fx0)
        if abs(fx0) <= 1e-8:
            return x0
    
    # Strategy 1: Newton's method with adaptive finite-difference derivative
    if check_deadline():
        x = x0
        for _ in range(100):
            if not check_deadline():
                break
            fx = safe_f(x)
            if fx is None:
                break
            if abs(fx) <= 1e-8:
                return x
            update_best(x, fx)
            
            # Adaptive step size for derivative
            h = 1e-5 * max(1.0, abs(x))
            fx_plus = safe_f(x + h)
            if fx_plus is None:
                break
            
            # Central difference if possible for better accuracy
            fx_minus = safe_f(x - h)
            if fx_minus is not None:
                d = (fx_plus - fx_minus) / (2 * h)
            else:
                d = (fx_plus - fx) / h
            
            if d == 0 or not is_finite(d) or abs(d) < 1e-15:
                break
            x_new = x - fx / d
            if not is_finite(x_new):
                break
            # Damping if step is too large
            if abs(x_new - x) > 5 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
            x = x_new
    
    # Strategy 2: Secant method (uses two points)
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        x1 = x0
        x2 = x0 + 0.1 * max(1.0, abs(x0))
        fx1 = safe_f(x1)
        fx2 = safe_f(x2)
        if fx1 is not None and fx2 is not None and fx1 != fx2:
            for _ in range(100):
                if not check_deadline():
                    break
                if abs(fx2) <= 1e-8:
                    return x2
                update_best(x2, fx2)
                if abs(fx1) <= 1e-8:
                    return x1
                update_best(x1, fx1)
                
                denom = fx2 - fx1
                if denom == 0 or not is_finite(denom):
                    break
                x_new = x2 - fx2 * (x2 - x1) / denom
                if not is_finite(x_new):
                    break
                # Keep points distinct
                if abs(x_new - x1) < 1e-12:
                    x_new = x1 + 1e-6
                if abs(x_new - x2) < 1e-12:
                    x_new = x2 + 1e-6
                
                x1, fx1 = x2, fx2
                x2 = x_new
                fx2 = safe_f(x2)
                if fx2 is None:
                    break
    
    # Strategy 3: Bisection with bracketing search
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        # Try to find a bracket around x0
        a, b = None, None
        fa, fb = None, None
        
        # Expand search range
        scale = 1.0
        for _ in range(50):
            if not check_deadline():
                break
            a_try = x0 - scale
            b_try = x0 + scale
            
            fa_try = safe_f(a_try)
            fb_try = safe_f(b_try)
            
            if fa_try is not None and fb_try is not None:
                if fa_try * fb_try <= 0:
                    a, b, fa, fb = a_try, b_try, fa_try, fb_try
                    break
                # Check if either endpoint is good enough
                if abs(fa_try) <= 1e-8:
                    return a_try
                if abs(fb_try) <= 1e-8:
                    return b_try
            
            scale *= 2
            if scale > 1e6:
                break
        
        if a is not None and b is not None:
            for _ in range(200):
                if not check_deadline():
                    break
                mid = (a + b) / 2.0
                fmid = safe_f(mid)
                if fmid is None:
                    break
                if abs(fmid) <= 1e-8:
                    return mid
                update_best(mid, fmid)
                
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
    
    # Strategy 4: Global search with multiple starting points
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        # Try several starting points
        offsets = [0.01, -0.01, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0,
                   0.001, -0.001, 100.0, -100.0]
        for offset in offsets:
            if not check_deadline():
                break
            x_start = x0 + offset * max(1.0, abs(x0))
            fx_start = safe_f(x_start)
            if fx_start is None:
                continue
            if abs(fx_start) <= 1e-8:
                return x_start
            update_best(x_start, fx_start)
            
            # Quick Newton from this point
            x = x_start
            for _ in range(50):
                if not check_deadline():
                    break
                fx = safe_f(x)
                if fx is None:
                    break
                if abs(fx) <= 1e-8:
                    return x
                h = 1e-5 * max(1.0, abs(x))
                fx_plus = safe_f(x + h)
                if fx_plus is None:
                    break
                d = (fx_plus - fx) / h
                if d == 0 or not is_finite(d) or abs(d) < 1e-15:
                    break
                x_new = x - fx / d
                if not is_finite(x_new):
                    break
                x = x_new
    
    # Return best found
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
