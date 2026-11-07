"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    tol = 1e-8
    
    x = float(x0)
    
    # Initial evaluation
    try:
        fx = f(x)
    except:
        return x
    if not math.isfinite(fx):
        return x
    if abs(fx) <= tol:
        return x
    
    # Use secant method with bisection safeguard
    # Start with a small perturbation to get a second point
    h = 1e-4 * max(1.0, abs(x))
    x_prev = x
    f_prev = fx
    x_curr = x + h
    
    # Try to find a bracket by expanding
    a, b, fa, fb = None, None, None, None
    for attempt in range(20):
        try:
            fc = f(x_curr)
        except:
            break
        if not math.isfinite(fc):
            break
        if abs(fc) <= tol:
            return x_curr
        
        # Check if we have a sign change (bracket)
        if fa is None:
            a, b, fa, fb = x, x_curr, fx, fc
        else:
            if fa * fc <= 0:
                a, b, fa, fb = x, x_curr, fa, fc
                break
            elif fb * fc <= 0:
                a, b, fa, fb = x_curr, x + h * (attempt + 1), fc, f(x + h * (attempt + 1)) if attempt + 1 < 20 else fc
                break
            else:
                # Expand search range
                x_curr = x + h * (attempt + 2)
                if attempt < 19:
                    try:
                        fc2 = f(x_curr)
                    except:
                        break
                    if not math.isfinite(fc2):
                        break
                    fb = fc2
                    b = x_curr
    
    # If we have a bracket, use bisection + secant hybrid
    if a is not None and b is not None and fa * fb <= 0:
        # Bisection with secant acceleration
        for _ in range(100):
            if time.time() > deadline:
                break
            mid = 0.5 * (a + b)
            try:
                fm = f(mid)
            except:
                break
            if not math.isfinite(fm):
                break
            if abs(fm) <= tol:
                return mid
            
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            
            # Try secant step if it stays within bracket
            if b != a:
                x_sec = a - fa * (b - a) / (fb - fa)
                if a < x_sec < b:
                    try:
                        fs = f(x_sec)
                    except:
                        fs = None
                    if fs is not None and math.isfinite(fs):
                        if abs(fs) <= tol:
                            return x_sec
                        if fa * fs <= 0:
                            b = x_sec
                            fb = fs
                        else:
                            a = x_sec
                            fa = fs
    
    # Polish pass: use Newton's method with high precision if we're close
    x_best = x
    f_best = fx
    if a is not None and b is not None:
        # Use midpoint as starting point for polish
        x = 0.5 * (a + b)
    else:
        # Use secant method without bracket
        x_prev = x
        f_prev = fx
        for _ in range(50):
            if time.time() > deadline:
                break
            x_curr = x + 1e-4 * max(1.0, abs(x))
            try:
                fc = f(x_curr)
            except:
                break
            if not math.isfinite(fc):
                break
            if abs(fc) <= tol:
                return x_curr
            if abs(fc) < abs(f_prev):
                x_best = x_curr
                f_best = fc
            
            # Secant step
            if fc != f_prev:
                x_new = x_curr - fc * (x_curr - x_prev) / (fc - f_prev)
                if math.isfinite(x_new):
                    x_prev = x_curr
                    f_prev = fc
                    x = x_new
            else:
                break
    
    # Final high-precision polish with Newton's method
    for _ in range(20):
        if time.time() > deadline:
            break
        try:
            fx = f(x)
        except:
            break
        if not math.isfinite(fx):
            break
        if abs(fx) <= tol:
            return x
        
        # Central difference for derivative
        h = 1e-7 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break
        
        d = (f_xp - f_xm) / (2.0 * h)
        if abs(d) < 1e-14:
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        
        # Damping
        step = x_new - x
        if abs(step) > 1.0:
            x_new = x + (1.0 if step > 0 else -1.0)
        
        x = x_new
    
    return x
# EVOLVE-BLOCK-END
