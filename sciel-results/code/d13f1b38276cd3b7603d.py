"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    
    if not math.isfinite(best_x) or not math.isfinite(best_fx):
        return x0
    
    if best_abs <= 1e-8:
        return x
    
    # Phase 1: Secant method to get close (robust, no derivative needed)
    # Use 2 points for secant
    h0 = 1e-4 * max(1.0, abs(x))
    x1 = x + h0
    f1 = f(x1)
    best_fx2 = f1
    if abs(f1) < best_abs:
        best_x = x1
        best_abs = abs(f1)
    
    if not math.isfinite(f1):
        return best_x if best_abs <= 1e-8 else x
    
    # Secant iterations to get into Newton's basin
    x_prev = x
    f_prev = best_fx
    x_curr = x1
    f_curr = f1
    best_x = x_curr if abs(f_curr) < best_abs else best_x
    best_abs = min(best_abs, abs(f_curr))
    
    for i in range(8):
        if time.time() > deadline:
            break
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_curr) < best_abs:
            best_x = x_curr
            best_abs = abs(f_curr)
        
        if abs(f_curr - f_prev) < 1e-30:
            break
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        if not math.isfinite(x_next):
            break
        
        # Damping if step is too large
        step = abs(x_next - x_curr)
        if step > 10.0 * max(1.0, abs(x_curr)):
            x_next = x_curr + math.copysign(10.0 * max(1.0, abs(x_curr)), x_next - x_curr)
        
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f(x_next)
        
        if not math.isfinite(f_curr):
            x_curr, f_curr = x_prev, f_prev
            break
    
    if abs(f_curr) <= 1e-8:
        return x_curr
    
    # Phase 2: Newton's method with finite-difference derivative
    x = x_curr
    fx = f_curr
    
    for i in range(30):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-8:
            return x
        if abs(fx) < best_abs:
            best_x = x
            best_abs = abs(fx)
        
        # Adaptive step for finite difference
        h = 1e-7 * max(1.0, abs(x))
        fx_h = f(x + h)
        
        if not math.isfinite(fx_h):
            h = 1e-5 * max(1.0, abs(x))
            fx_h = f(x + h)
            if not math.isfinite(fx_h):
                break
        
        d = (fx_h - fx) / h
        if abs(d) < 1e-30:
            break
        
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
        
        # Damping
        step = abs(x_new - x)
        if step > 5.0 * max(1.0, abs(x)):
            x_new = x + math.copysign(5.0 * max(1.0, abs(x)), x_new - x)
        
        x = x_new
        fx = f(x)
        
        if not math.isfinite(fx):
            break
    
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 3: Polish pass with tighter tolerance
    # Try a few more Newton steps with tighter check
    for i in range(10):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-10:
            return x
        if abs(fx) < best_abs:
            best_x = x
            best_abs = abs(fx)
        
        h = 1e-8 * max(1.0, abs(x))
        fx_h = f(x + h)
        if not math.isfinite(fx_h):
            break
        d = (fx_h - fx) / h
        if abs(d) < 1e-30:
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
        fx = f(x)
        if not math.isfinite(fx):
            break
    
    if abs(fx) <= 1e-8:
        return x
    
    # Return best found
    if best_abs <= 1e-8:
        return best_x
    
    # Last resort: return x if it's finite, else best_x, else x0
    if math.isfinite(x) and abs(fx) < best_abs:
        return x
    if math.isfinite(best_x):
        return best_x
    return x0
# EVOLVE-BLOCK-END
