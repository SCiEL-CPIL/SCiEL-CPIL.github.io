"""Robust scalar root finder with multiple strategies and deadline guard."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a multi-strategy approach: Newton/secant with safeguarding,
    bisection bracketing, and fallback global search.
    Respects a wall-clock deadline.
    """
    DEADLINE = 145.0  # Safety margin before 150s kill
    START_TIME = time.time()
    TOL = 1e-8
    
    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE
    
    def is_finite(x):
        return math.isfinite(x)
    
    def safe_f(x):
        """Evaluate f(x) safely, returning None on failure."""
        try:
            fx = f(x)
            if is_finite(fx):
                return fx
        except Exception:
            pass
        return None
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
    
    best_x = float(x0)
    best_fx = safe_f(best_x)
    if best_fx is None:
        best_fx = float('inf')
    
    if best_fx is not None and abs(best_fx) <= TOL:
        return best_x
    
    # Strategy 1: Newton's method with finite-difference derivative
    # Use a hybrid approach: Newton step if it stays in a reasonable range,
    # otherwise bisect.
    
    x = float(x0)
    fx = safe_f(x)
    if fx is None:
        return best_x
    update_best(x, fx)
    if abs(fx) <= TOL:
        return x
    
    # Try to establish a bracket around x0
    a = x
    fa = fx
    b = None
    fb = None
    
    # Expand bracket to find sign change
    step = max(1.0, abs(x) * 0.1)
    for direction in [1, -1]:
        for i in range(1, 20):
            if not check_deadline():
                break
            candidate = x + direction * step * (2 ** i)
            fc = safe_f(candidate)
            if fc is None:
                break
            if fa * fc < 0:
                if direction == 1:
                    b = candidate
                    fb = fc
                else:
                    b = candidate
                    fb = fc
                break
            if abs(fc) < abs(fa):
                # Move bracket closer
                a = candidate
                fa = fc
                break
        if b is not None:
            break
    
    # If we have a bracket, use bisection + Newton hybrid
    if b is not None and check_deadline():
        lo = min(a, b)
        hi = max(a, b)
        flo = safe_f(lo)
        fhi = safe_f(hi)
        
        for _ in range(100):
            if not check_deadline():
                break
            mid = (lo + hi) / 2.0
            fmid = safe_f(mid)
            if fmid is None:
                break
            update_best(mid, fmid)
            if abs(fmid) <= TOL:
                return mid
            
            if flo * fmid <= 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
            
            # Try Newton step from mid
            h = 1e-8 * max(1.0, abs(mid))
            fmid_plus = safe_f(mid + h)
            if fmid_plus is not None:
                d = (fmid_plus - fmid) / h
                if d != 0 and math.isfinite(d):
                    x_new = mid - fmid / d
                    if lo < x_new < hi:
                        f_new = safe_f(x_new)
                        if f_new is not None:
                            update_best(x_new, f_new)
                            if abs(f_new) <= TOL:
                                return x_new
                            if flo * f_new <= 0:
                                hi = x_new
                                fhi = f_new
                            else:
                                lo = x_new
                                flo = f_new
                    elif abs(x_new - mid) < (hi - lo) * 0.5:
                        # Newton step is within reasonable range
                        f_new = safe_f(x_new)
                        if f_new is not None:
                            update_best(x_new, f_new)
                            if abs(f_new) <= TOL:
                                return x_new
    
    # Strategy 2: Secant method / Newton with safeguarding
    if check_deadline():
        x1 = float(x0)
        fx1 = safe_f(x1)
        if fx1 is None:
            return best_x
        update_best(x1, fx1)
        if abs(fx1) <= TOL:
            return x1
        
        h = 1e-6 * max(1.0, abs(x1))
        x2 = x1 + h
        fx2 = safe_f(x2)
        if fx2 is None:
            return best_x
        update_best(x2, fx2)
        if abs(fx2) <= TOL:
            return x2
        
        for _ in range(100):
            if not check_deadline():
                break
            
            # Secant step
            if fx2 != fx1:
                x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            else:
                # Use Newton if secant fails
                d = (fx2 - fx1) / (x2 - x1)
                if d == 0:
                    break
                x_new = x2 - fx2 / d
            
            if not is_finite(x_new):
                break
            
            fx_new = safe_f(x_new)
            if fx_new is None:
                # Shrink step
                x_new = (x1 + x2) / 2.0
                fx_new = safe_f(x_new)
                if fx_new is None:
                    break
            
            update_best(x_new, fx_new)
            if abs(fx_new) <= TOL:
                return x_new
            
            # Keep the two most recent points
            x1 = x2
            fx1 = fx2
            x2 = x_new
            fx2 = fx_new
    
    # Strategy 3: Global search if local methods failed
    if check_deadline():
        # Search in expanding intervals
        base = abs(float(x0))
        if base == 0:
            base = 1.0
        
        for scale in [1, 2, 5, 10, 50, 100, 500, 1000]:
            if not check_deadline():
                break
            interval = base * scale
            n_points = 50
            prev_x = -interval
            prev_f = safe_f(prev_x)
            if prev_f is None:
                continue
            update_best(prev_x, prev_f)
            
            for i in range(1, n_points + 1):
                if not check_deadline():
                    break
                x_i = -interval + 2 * interval * i / n_points
                f_i = safe_f(x_i)
                if f_i is None:
                    prev_x = x_i
                    prev_f = f_i
                    continue
                update_best(x_i, f_i)
                
                if prev_f * f_i < 0:
                    # Found a bracket, use bisection
                    lo = prev_x
                    hi = x_i
                    flo = prev_f
                    fhi = f_i
                    for _ in range(60):
                        if not check_deadline():
                            break
                        mid = (lo + hi) / 2.0
                        fmid = safe_f(mid)
                        if fmid is None:
                            break
                        update_best(mid, fmid)
                        if abs(fmid) <= TOL:
                            return mid
                        if flo * fmid <= 0:
                            hi = mid
                            fhi = fmid
                        else:
                            lo = mid
                            flo = fmid
                    break
                prev_x = x_i
                prev_f = f_i
    
    # Return the best result found
    if best_fx is not None and abs(best_fx) <= TOL:
        return best_x
    
    # If we haven't found a root within tolerance, try one more Newton-like approach
    if check_deadline():
        x = best_x
        for _ in range(30):
            if not check_deadline():
                break
            fx = safe_f(x)
            if fx is None:
                break
            update_best(x, fx)
            if abs(fx) <= TOL:
                return x
            h = 1e-8 * max(1.0, abs(x))
            fx_plus = safe_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not is_finite(x_new):
                break
            x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
