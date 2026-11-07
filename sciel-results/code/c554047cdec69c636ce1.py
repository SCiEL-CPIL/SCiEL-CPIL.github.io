"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a hybrid secant/bisection method
    with safeguards and a time limit.
    """
    start_time = time.time()
    TIME_LIMIT = 140.0  # seconds, leaving margin for evaluator overhead
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT
    
    x0 = float(x0)
    
    # Try to find a bracketing interval [a, b] where f(a) and f(b) have opposite signs
    # Start with a small interval around x0 and expand if needed
    
    def find_bracket(f, x0):
        """Find an interval [a, b] where f(a)*f(b) < 0"""
        # Start with a small step
        step = max(1e-6, abs(x0) * 1e-6)
        
        # Try expanding in both directions
        for i in range(50):
            if not time_ok():
                return None
            
            # Try [x0 - step, x0 + step]
            a = x0 - step
            b = x0 + step
            fa = f(a)
            fb = f(b)
            
            if fa == 0:
                return (a, a)
            if fb == 0:
                return (b, b)
            if fa * fb < 0:
                return (a, b)
            
            # Expand the interval
            step *= 2.0
            if step > 1e10:
                break
        
        # Try asymmetric expansions
        step = max(1e-6, abs(x0) * 1e-6)
        for i in range(30):
            if not time_ok():
                return None
            
            # Try [x0, x0 + step]
            a = x0
            b = x0 + step
            fa = f(a)
            fb = f(b)
            if fa * fb < 0:
                return (a, b)
            
            # Try [x0 - step, x0]
            a = x0 - step
            b = x0
            fa = f(a)
            fb = f(b)
            if fa * fb < 0:
                return (a, b)
            
            step *= 2.0
            if step > 1e10:
                break
        
        return None
    
    # First, try secant method which is fast for smooth functions
    # Use a perturbed second point
    x = float(x0)
    fx = f(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    # Secant method with safeguards
    h = 1e-4 * max(1.0, abs(x))
    x_prev = x + h
    fx_prev = f(x_prev)
    
    if abs(fx_prev) <= 1e-8:
        return x_prev
    
    best_x = x
    best_fx = abs(fx)
    
    for i in range(40):
        if not time_ok():
            break
        
        # Check if we've found a good solution
        if abs(fx) < best_fx:
            best_fx = abs(fx)
            best_x = x
        
        if abs(fx) <= 1e-8:
            return x
        
        # Secant step
        if fx_prev != fx:
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
        else:
            break
        
        # Safeguard: if x_new is too far, use bisection-like step
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = (x + x_prev) / 2.0
        
        # Ensure x_new is finite
        if not math.isfinite(x_new):
            break
        
        x_prev = x
        fx_prev = fx
        x = x_new
        fx = f(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    # If secant didn't work well, try bisection with bracketing
    bracket = find_bracket(f, x0)
    if bracket is not None:
        a, b = bracket
        fa = f(a)
        fb = f(b)
        
        # Bisection
        for i in range(60):
            if not time_ok():
                break
            
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            
            mid = (a + b) / 2.0
            fm = f(mid)
            
            if abs(fm) <= 1e-8:
                return mid
            
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
    
    # Return the best we found
    if abs(f(best_x)) <= 1e-8:
        return best_x
    
    # Final attempt: try a few more secant steps from best_x
    x = best_x
    fx = f(x)
    for i in range(10):
        if not time_ok():
            break
        if abs(fx) <= 1e-8:
            return x
        
        h = 1e-5 * max(1.0, abs(x))
        x2 = x + h
        fx2 = f(x2)
        
        if fx2 != fx:
            x = x - fx * (x - x2) / (fx - fx2)
            fx = f(x)
    
    return x
# EVOLVE-BLOCK-END
