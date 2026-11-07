"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Finds a root of f starting near x0 using the Secant method.
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well under 120s limit
    
    x = float(x0)
    
    # Initial step to get a second point
    # Use a small relative step to start secant method
    h = 1e-4 * max(1.0, abs(x))
    x_prev = x
    x_curr = x + h
    
    try:
        f_prev = f(x_prev)
        f_curr = f(x_curr)
    except Exception:
        return x

    best_x = x_prev
    best_f = f_prev
    
    if abs(f_prev) <= 1e-8:
        return x_prev
    if abs(f_curr) <= 1e-8:
        return x_curr
        
    if abs(f_curr) < abs(f_prev):
        best_x = x_curr
        best_f = f_curr

    # Secant method loop
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Check for zero slope
        if f_curr == f_prev:
            # Degenerate case, try a different step
            x_curr = x_prev + 1e-3 * max(1.0, abs(x_prev))
            try:
                f_curr = f(x_curr)
            except Exception:
                break
            if abs(f_curr) <= 1e-8:
                return x_curr
            if abs(f_curr) < abs(best_f):
                best_x = x_curr
                best_f = f_curr
            # Update prev to be the point with larger f to avoid division by zero in next step?
            # Actually, if f_curr == f_prev, the secant line is horizontal.
            # We should restart or use a different method. 
            # For simplicity, we'll just break and return best.
            break
        
        # Secant step
        denom = f_curr - f_prev
        dx = (x_curr - x_prev) * f_curr / denom
        x_new = x_curr - dx
        
        # Safety check: if x_new is not finite, break
        if not math.isfinite(x_new):
            break
            
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        if abs(f_new) < abs(best_f):
            best_x = x_new
            best_f = f_new
            
        # Shift points
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new

    return best_x
# EVOLVE-BLOCK-END
