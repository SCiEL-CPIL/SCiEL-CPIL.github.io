"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 100.0  # Safety margin before 150s kill
    x = float(x0)
    
    # Check initial point
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try Newton's method with adaptive step size
    for _ in range(50):
        if time.monotonic() > deadline:
            break
        if abs(fx) <= 1e-10:
            return x
        
        # Adaptive finite difference step
        h = 1e-5 * max(1.0, abs(x))
        fx_h = f(x + h)
        d = (fx_h - fx) / h
        
        if abs(d) < 1e-15:
            # Derivative too small, try bisection-like step
            x = x - math.copysign(1e-4, fx)
            fx = f(x)
            continue
            
        x_new = x - fx / d
        
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x - math.copysign(0.1 * max(1.0, abs(x)), fx)
        
        x = x_new
        fx = f(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    # Fallback: try a few more points with different strategies
    for offset in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
        if time.monotonic() > deadline:
            break
        x_try = x + offset
        fx_try = f(x_try)
        if abs(fx_try) <= 1e-8:
            return x_try
        
        # Quick Newton step from this point
        for _ in range(10):
            if time.monotonic() > deadline:
                break
            if abs(fx_try) <= 1e-10:
                return x_try
            h = 1e-5 * max(1.0, abs(x_try))
            d = (f(x_try + h) - fx_try) / h
            if abs(d) < 1e-15:
                break
            x_try = x_try - fx_try / d
            fx_try = f(x_try)
    
    return x
# EVOLVE-BLOCK-END
