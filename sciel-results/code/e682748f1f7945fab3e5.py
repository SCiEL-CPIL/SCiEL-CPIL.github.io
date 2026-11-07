import time
import math

_DEADLINE = 140.0  # safety margin before 150s kill

def find_root(f, x0):
    t0 = time.monotonic()
    x = float(x0)
    
    # Evaluate initial f(x)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Store for reuse
    prev_x = x
    prev_fx = fx
    
    # Phase 1: Newton's method with forward-difference derivative
    # Reuse f(x) from previous iteration to minimize calls
    for i in range(50):
        if time.monotonic() - t0 > _DEADLINE:
            break
        if abs(fx) <= 1e-8:
            return x
        
        # Central difference for better accuracy (2 extra calls)
        # Actually, use forward diff to save calls: 1 extra call
        h = 1e-7 * max(1.0, abs(x))
        fxh = f(x + h)
        d = (fxh - fx) / h
        
        if abs(d) < 1e-30:
            # Derivative near zero, try secant step
            if abs(prev_x - x) > 1e-15:
                d_sec = (fx - prev_fx) / (x - prev_x)
                if abs(d_sec) > 1e-30:
                    x_new = x - fx / d_sec
                else:
                    # Nudge
                    x_new = x + 1e-6
            else:
                x_new = x + 1e-6
        else:
            x_new = x - fx / d
            # Damping: if step is too large, reduce it
            if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
                x_new = x + 10.0 * max(1.0, abs(x)) * (1 if x_new > x else -1)
        
        prev_x = x
        prev_fx = fx
        x = x_new
        fx = f(x)
    
    # Phase 2: High-precision polish
    for i in range(30):
        if time.monotonic() - t0 > _DEADLINE:
            break
        if abs(fx) <= 1e-12:
            return x
        
        h = 1e-8 * max(1.0, abs(x))
        fxh = f(x + h)
        fxl = f(x - h)
        d = (fxh - fxl) / (2 * h)
        
        if abs(d) < 1e-30:
            break
        
        x = x - fx / d
        fx = f(x)
    
    # Return best result
    if abs(fx) <= 1e-8:
        return x
    # Try one more secant step
    if abs(prev_x - x) > 1e-15:
        d = (fx - prev_fx) / (x - prev_x)
        if abs(d) > 1e-30:
            x = x - fx / d
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
    
    return x
