"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    MAX_CALLS = 490
    DEADLINE = time.time() + 145.0
    calls = [0]
    
    def feval(x):
        if calls[0] >= MAX_CALLS or time.time() > DEADLINE:
            return None
        calls[0] += 1
        try:
            return f(x)
        except Exception:
            return None
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    fx = feval(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-10:
        return x
    
    best_x = x
    best_fx = fx
    
    # Newton's method with finite-difference derivative
    x_prev = None
    fx_prev = None
    for _ in range(120):
        if time.time() > DEADLINE - 5.0 or calls[0] >= MAX_CALLS:
            break
        if abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        if abs(fx) <= 1e-10:
            return x
        
        h = 1e-6 * max(1.0, abs(x))
        fxh = feval(x + h)
        if fxh is None:
            break
        d = (fxh - fx) / h
        if d == 0 or not math.isfinite(d):
            # Try secant step if we have a previous point
            if x_prev is not None and fx_prev is not None and fx_prev != fx:
                x_new = x - fx * (x - x_prev) / (fx - fx_prev)
                if math.isfinite(x_new):
                    x = x_new
                else:
                    break
            else:
                break
        else:
            x_new = x - fx / d
            if not math.isfinite(x_new):
                # Damping: take half step
                x_new = x - 0.5 * fx / d
            x = x_new
        
        x_prev = x
        fx = feval(x)
        if fx is None:
            break
        fx_prev = fx
    
    # If we haven't converged, try a few secant iterations
    if abs(best_fx) > 1e-10 and x_prev is not None and fx_prev is not None:
        for _ in range(30):
            if time.time() > DEADLINE - 5.0 or calls[0] >= MAX_CALLS:
                break
            if abs(fx) < abs(best_fx):
                best_x = x
                best_fx = fx
            if abs(fx) <= 1e-10:
                return x
            if fx == fx_prev:
                break
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
            if not math.isfinite(x_new):
                break
            x_prev = x
            fx_prev = fx
            x = x_new
            fx = feval(x)
            if fx is None:
                break
    
    return best_x
# EVOLVE-BLOCK-END
