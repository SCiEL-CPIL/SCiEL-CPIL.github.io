"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def time_ok():
        return time.time() < deadline
    
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
    
    # Try to find a better starting point by sampling nearby
    # Use a small step for derivative estimation
    h = 1e-6 * max(1.0, abs(x))
    
    best_x = x
    best_fabs = abs(fx)
    
    # Sample a few nearby points to find a good starting point
    for step in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
        if not time_ok():
            break
        for sign in [1, -1]:
            x_try = x + sign * step * max(1.0, abs(x))
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try) and abs(fx_try) < best_fabs:
                    best_x = x_try
                    best_fabs = abs(fx_try)
                    if best_fabs <= 1e-8:
                        return x_try
            except Exception:
                pass
    
    if best_fabs <= 1e-8:
        return best_x
    
    # Use Newton's method with safeguarding from best_x
    x = best_x
    fx = f(x) if abs(fx) > 1e-8 else fx  # Reuse if possible
    
    for iteration in range(100):
        if not time_ok():
            break
        
        try:
            fx = f(x)
        except Exception:
            break
        
        if not math.isfinite(fx):
            break
        
        if abs(fx) <= 1e-8:
            return x
        
        # Compute derivative using central difference
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
        except Exception:
            break
        
        if not math.isfinite(fx_plus) or not math.isfinite(fx_minus):
            break
        
        d = (fx_plus - fx_minus) / (2.0 * h)
        
        if not math.isfinite(d) or abs(d) < 1e-15:
            # If derivative is too small, try a larger step
            h2 = 1e-4 * max(1.0, abs(x))
            try:
                fx_plus2 = f(x + h2)
                fx_minus2 = f(x - h2)
                d = (fx_plus2 - fx_minus2) / (2.0 * h2)
            except Exception:
                break
            
            if not math.isfinite(d) or abs(d) < 1e-15:
                break
        
        # Newton step
        x_new = x - fx / d
        
        # Safeguard: if step is too large, dampen it
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
        
        if not math.isfinite(x_new):
            break
        
        x = x_new
    
    return x
# EVOLVE-BLOCK-END
