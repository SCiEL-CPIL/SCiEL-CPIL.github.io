"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline with 10s safety margin
    x = float(x0)
    
    # Keep track of best x found so far
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    
    # Try Newton's method with safeguards
    for iteration in range(80):
        if time.time() > deadline:
            break
        
        fx = f(x)
        abs_fx = abs(fx)
        
        if abs_fx <= 1e-12:
            return x
        
        if abs_fx < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs_fx
        
        # Finite difference derivative with adaptive step
        h = 1e-7 * max(1.0, abs(x))
        if h < 1e-12:
            h = 1e-12
        if h > 1e3:
            h = 1e3
        
        try:
            fx_plus = f(x + h)
            d = (fx_plus - fx) / h
        except Exception:
            d = 0.0
        
        if d == 0 or not _is_finite(d):
            # Try a different step size
            h2 = 1e-5 * max(1.0, abs(x))
            try:
                fx_plus2 = f(x + h2)
                d = (fx_plus2 - fx) / h2
            except Exception:
                d = 0.0
        
        if d == 0 or not _is_finite(d):
            break
        
        # Newton step with safeguard
        step = fx / d
        new_x = x - step
        
        # Check if new point is reasonable
        if not _is_finite(new_x):
            break
        
        # Limit step size to prevent divergence
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
            new_x = x - step
        
        # Check if function value is getting worse
        try:
            fx_new = f(new_x)
            if not _is_finite(fx_new):
                # Halve the step
                step = step / 2.0
                new_x = x - step
                fx_new = f(new_x)
                if not _is_finite(fx_new):
                    break
        except Exception:
            break
        
        x = new_x
    
    # If Newton didn't converge, try secant method
    if best_abs_fx > 1e-12 and time.time() < deadline:
        x = _secant_root(f, x0, deadline, best_x, best_fx)
        if x is not None and _is_finite(x):
            try:
                fx_check = f(x)
                if _is_finite(fx_check) and abs(fx_check) < best_abs_fx:
                    best_x = x
                    best_abs_fx = abs(fx_check)
            except Exception:
                pass
    
    return best_x

def _is_finite(val):
    try:
        return val == val and val != float('inf') and val != float('-inf')
    except Exception:
        return False

def _secant_root(f, x0, deadline, best_x, best_fx):
    """Secant method with safeguards."""
    x0 = float(x0)
    x1 = x0 + 1e-4 * max(1.0, abs(x0))
    
    try:
        f0 = f(x0)
        f1 = f(x1)
    except Exception:
        return None
    
    if not _is_finite(f0) or not _is_finite(f1):
        return None
    
    for _ in range(30):
        if time.time() > deadline:
            break
        
        if abs(f1) <= 1e-12:
            return x1
        
        if abs(f0) <= 1e-12:
            return x0
        
        denom = f1 - f0
        if denom == 0 or not _is_finite(denom):
            break
        
        x2 = x1 - f1 * (x1 - x0) / denom
        
        if not _is_finite(x2):
            break
        
        # Limit step
        max_step = 10.0 * max(1.0, abs(x1))
        if abs(x2 - x1) > max_step:
            x2 = x1 + max_step if x2 > x1 else x1 - max_step
        
        try:
            f2 = f(x2)
            if not _is_finite(f2):
                break
        except Exception:
            break
        
        x0, f0 = x1, f1
        x1, f1 = x2, f2
    
    return x1
# EVOLVE-BLOCK-END
