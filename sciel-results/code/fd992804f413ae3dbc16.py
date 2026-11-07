"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Hybrid secant/bisection root finder with:
    - Initial bracketing search
    - Secant method for fast convergence when possible
    - Bisection fallback for robustness
    - Time guard to stay well under the 150s limit
    - Final polish pass with tighter tolerance
    """
    start_time = time.time()
    time_limit = 12.0  # Safety margin: 12s per equation max (3 equations * 12s = 36s << 150s)
    
    def elapsed():
        return time.time() - start_time
    
    x = float(x0)
    
    # First, evaluate at x0
    try:
        fx = f(x)
    except Exception:
        return x
    
    if not math.isfinite(fx):
        return x
    
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Start with a small step and expand
    a = x
    fa = fx
    step = max(1e-6, abs(x) * 1e-6)
    
    b = None
    fb = None
    
    # Search for a bracket by expanding in both directions
    for i in range(50):
        if elapsed() > time_limit * 0.1:
            break
        try:
            # Try expanding to the right
            x_right = x + step * (i + 1)
            fx_right = f(x_right)
            if math.isfinite(fx_right):
                if fa * fx_right < 0:
                    b = x_right
                    fb = fx_right
                    break
                # Check if this is better
                if abs(fx_right) < abs(fa):
                    a = x_right
                    fa = fx_right
            
            # Try expanding to the left
            x_left = x - step * (i + 1)
            fx_left = f(x_left)
            if math.isfinite(fx_left):
                if fa * fx_left < 0:
                    b = x_left
                    fb = fx_left
                    break
                if abs(fx_left) < abs(fa):
                    a = x_left
                    fa = fx_left
        except Exception:
            break
        
        step *= 2.0
    
    # If we have a bracket, use bisection + secant hybrid
    if b is not None and fb is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Hybrid secant/bisection
        x_prev = a
        fx_prev = fa
        
        for i in range(80):
            if elapsed() > time_limit * 0.8:
                break
            
            # Check if we've converged
            if abs(fx) <= 1e-8:
                break
            
            # Secant step
            if fx != fx_prev and (b - a) > 1e-15:
                x_secant = x - fx * (x - x_prev) / (fx - fx_prev)
            else:
                x_secant = (a + b) / 2.0
            
            # Check if secant step is within bracket
            if a < x_secant < b:
                x_next = x_secant
            else:
                x_next = (a + b) / 2.0
            
            try:
                fx_next = f(x_next)
            except Exception:
                x_next = (a + b) / 2.0
                try:
                    fx_next = f(x_next)
                except Exception:
                    break
            
            if not math.isfinite(fx_next):
                x_next = (a + b) / 2.0
                try:
                    fx_next = f(x_next)
                except Exception:
                    break
            
            # Update bracket
            if fa * fx_next <= 0:
                b = x_next
                fb = fx_next
            else:
                a = x_next
                fa = fx_next
            
            x_prev = x
            fx_prev = fx
            x = x_next
            fx = fx_next
            
            if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
                break
        
        # Final polish: use Newton-like refinement if possible
        if elapsed() < time_limit * 0.95:
            for _ in range(10):
                if abs(fx) <= 1e-12:
                    break
                h = 1e-8 * max(1.0, abs(x))
                try:
                    d = (f(x + h) - fx) / h
                except Exception:
                    break
                if d == 0 or not math.isfinite(d):
                    break
                x_new = x - fx / d
                if not math.isfinite(x_new):
                    break
                try:
                    fx_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(fx_new):
                    break
                x = x_new
                fx = fx_new
                if abs(x - x_new) < 1e-15 * max(1.0, abs(x)):
                    break
    else:
        # No bracket found: use secant method with finite difference derivative
        # Keep track of best x found
        best_x = x
        best_fx = fx
        
        x_prev = x
        fx_prev = fx
        
        for i in range(80):
            if elapsed() > time_limit * 0.8:
                break
            
            if abs(fx) <= 1e-8:
                best_x = x
                best_fx = fx
                break
            
            # Compute derivative via finite difference
            h = 1e-6 * max(1.0, abs(x))
            try:
                d = (f(x + h) - fx) / h
            except Exception:
                break
            
            if not math.isfinite(d) or d == 0:
                # Try secant step instead
                if fx != fx_prev and (x - x_prev) != 0:
                    x_new = x - fx * (x - x_prev) / (fx - fx_prev)
                else:
                    break
            else:
                x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
            
            # Damping: if step is too large, reduce it
            if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
                x_new = x + 0.5 * math.copysign(1.0, x_new - x) * min(abs(x_new - x), 10.0 * max(1.0, abs(x)))
            
            try:
                fx_new = f(x_new)
            except Exception:
                break
            
            if not math.isfinite(fx_new):
                break
            
            # Keep track of best
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            
            x_prev = x
            fx_prev = fx
            x = x_new
            fx = fx_new
            
            if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
                break
        
        # Final polish
        if elapsed() < time_limit * 0.95 and abs(best_fx) <= 1e-6:
            x = best_x
            fx = best_fx
            for _ in range(10):
                if abs(fx) <= 1e-12:
                    break
                h = 1e-8 * max(1.0, abs(x))
                try:
                    d = (f(x + h) - fx) / h
                except Exception:
                    break
                if d == 0 or not math.isfinite(d):
                    break
                x_new = x - fx / d
                if not math.isfinite(x_new):
                    break
                try:
                    fx_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(fx_new):
                    break
                x = x_new
                fx = fx_new
                if abs(x - x_new) < 1e-15 * max(1.0, abs(x)):
                    break
    
    # Return the best x found
    if abs(fx) <= 1e-8:
        return x
    
    # If we didn't converge, return the best we have
    if 'best_x' in locals() and 'best_fx' in locals():
        if abs(best_fx) <= 1e-8:
            return best_x
        # Return x if it's better, else best_x
        if abs(fx) < abs(best_fx):
            return x
        return best_x
    
    return x

# EVOLVE-BLOCK-END
