"""Robust scalar root finder with efficient Newton/secant and bisection fallback."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses Newton's method with adaptive step, secant fallback, and bisection.
    Respects a wall-clock deadline.
    """
    DEADLINE = 148.0  # Safety margin before 150s kill
    START_TIME = time.time()
    
    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE
    
    def is_finite(x):
        return math.isfinite(x)
    
    def safe_f(x):
        """Evaluate f(x) safely, returning None on failure."""
        try:
            fx = f(x)
            if is_finite(fx):
                return fx
        except:
            pass
        return None
    
    def is_good(x, fx):
        """Check if x is a good root."""
        return fx is not None and abs(fx) <= 1e-8
    
    # Track best solution found so far
    best_x = None
    best_fx = None
    best_abs_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx, best_abs_fx
        if fx is not None and is_finite(x) and abs(fx) < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs(fx)
            if abs(fx) <= 1e-8:
                return True
        return False
    
    # Initial evaluation
    fx0 = safe_f(x0)
    if fx0 is not None:
        update_best(x0, fx0)
        if abs(fx0) <= 1e-8:
            return x0
    
    x = float(x0)
    
    # Strategy 1: Newton's method with adaptive step and secant fallback
    x_prev = None
    fx_prev = None
    
    for iteration in range(200):
        if not check_deadline():
            break
        
        fx = safe_f(x)
        if fx is None:
            break
        
        if update_best(x, fx):
            return x
        
        # Compute derivative using central difference if possible
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = safe_f(x + h)
        fx_minus = safe_f(x - h)
        
        if fx_plus is not None and fx_minus is not None:
            d = (fx_plus - fx_minus) / (2 * h)
        elif fx_plus is not None:
            d = (fx_plus - fx) / h
        elif fx_minus is not None:
            d = (fx - fx_minus) / h
        else:
            # Fall back to secant method
            if x_prev is not None and fx_prev is not None and fx_prev != fx:
                d = (fx - fx_prev) / (x - x_prev)
            else:
                break
        
        if d == 0 or not is_finite(d):
            break
        
        # Newton step with damping
        step = fx / d
        if not is_finite(step):
            break
        
        # Damp large steps
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step * (1 if step > 0 else -1)
        
        x_new = x - step
        if not is_finite(x_new):
            break
        
        x_prev = x
        fx_prev = fx
        x = x_new
    
    # Strategy 2: Secant method with multiple starting points
    if check_deadline():
        offsets = [1e-3, 1e-2, 1e-1, 1.0, 10.0, -1e-3, -1e-2, -1e-1, -1.0, -10.0]
        for offset in offsets:
            if not check_deadline():
                break
            
            x1 = x0
            x2 = x0 + offset * max(1.0, abs(x0))
            fx1 = safe_f(x1)
            fx2 = safe_f(x2)
            
            if fx1 is None or fx2 is None:
                continue
            
            if update_best(x1, fx1):
                return x1
            if update_best(x2, fx2):
                return x2
            
            for _ in range(100):
                if not check_deadline():
                    break
                
                if fx1 == fx2:
                    break
                
                x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
                if not is_finite(x_new):
                    break
                
                x1 = x2
                fx1 = fx2
                x2 = x_new
                fx2 = safe_f(x2)
                
                if fx2 is None:
                    break
                
                if update_best(x2, fx2):
                    return x2
    
    # Strategy 3: Bisection with bracketing
    if check_deadline():
        # Try to find a bracket
        a = x0 - 1.0
        b = x0 + 1.0
        fa = safe_f(a)
        fb = safe_f(b)
        
        # Expand bracket if needed
        scale = 1.0
        while check_deadline() and scale < 1e6:
            if fa is not None and fb is not None and fa * fb < 0:
                break
            scale *= 2.0
            a = x0 - scale
            b = x0 + scale
            fa = safe_f(a)
            fb = safe_f(b)
        
        if fa is not None and fb is not None and fa * fb < 0:
            # Bisection
            for iteration in range(200):
                if not check_deadline():
                    break
                
                mid = (a + b) / 2.0
                fm = safe_f(mid)
                
                if fm is None:
                    break
                
                if update_best(mid, fm):
                    return mid
                
                if fa * fm < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
    
    # Return best solution found
    if best_x is not None:
        return best_x
    
    # Fallback: return x0 if nothing better found
    return float(x0)
# EVOLVE-BLOCK-END
