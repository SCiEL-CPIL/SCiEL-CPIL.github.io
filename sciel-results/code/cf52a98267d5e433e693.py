import math
import time

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    max_calls = 480
    
    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= max_calls:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except:
            return float('nan')
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_abs_f:
                best_abs_f = abs_val
                best_x = x
        return val

    def time_check():
        return time.time() - start_time < TIME_LIMIT - 5.0

    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # 1. Find Bracket - try multiple strategies
    a, b, fa, fb = None, None, None, None
    found_bracket = False
    
    # Strategy: Try expanding in both directions with adaptive steps
    base_step = 1.0
    if abs(x) > 1e6: base_step = 1e6
    elif abs(x) > 1e3: base_step = 1e3
    elif abs(x) > 1: base_step = 1.0
    else: base_step = 0.1
    
    # Try positive direction first
    step = base_step * 0.1
    for i in range(1, 15):
        if not time_check() or calls >= max_calls - 50:
            break
        xp = x + step
        fp = _f(xp)
        if math.isnan(fp): break
        if abs(fp) <= 1e-8: return xp
        if fx * fp < 0:
            a, b, fa, fb = x, xp, fx, fp
            found_bracket = True
            break
        step *= 1.5
        
    if not found_bracket:
        # Try negative direction
        step = base_step * 0.1
        for i in range(1, 15):
            if not time_check() or calls >= max_calls - 50:
                break
            xn = x - step
            fn = _f(xn)
            if math.isnan(fn): break
            if abs(fn) <= 1e-8: return xn
            if fx * fn < 0:
                a, b, fa, fb = xn, x, fn, fx
                found_bracket = True
                break
            step *= 1.5

    if found_bracket:
        # 2. Refine using Guarded Secant / Illinois
        for i in range(80):
            if not time_check() or calls >= max_calls - 10:
                break
            if abs(fa) <= 1e-8: return a
            if abs(fb) <= 1e-8: return b
            
            # Try Secant
            if abs(fa - fb) > 1e-30:
                m = (a * fb - b * fa) / (fb - fa)
            else:
                m = (a + b) / 2
                
            if a < m < b:
                fm = _f(m)
                if math.isnan(fm): break
                if abs(fm) <= 1e-8: return m
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            else:
                # Bisection
                m = (a + b) / 2
                fm = _f(m)
                if math.isnan(fm): break
                if abs(fm) <= 1e-8: return m
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            
        # Return the better endpoint
        return a if abs(fa) < abs(fb) else b

    # 3. If no bracket found, do local minimization of |f(x)|
    # Use a simple expanding search for minimum |f|
    if time_check() and calls < max_calls - 30:
        # Try a few candidate points around x0
        candidates = [x]
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            for sign in [1, -1]:
                if abs(x) > 1e6:
                    step = 1e6 * scale
                elif abs(x) > 1e3:
                    step = 1e3 * scale
                elif abs(x) > 1:
                    step = scale
                else:
                    step = scale * 0.1
                candidates.append(x + sign * step)
        
        # Evaluate candidates and keep the best
        for cx in candidates:
            if not time_check() or calls >= max_calls - 20:
                break
            cf = _f(cx)
            if math.isnan(cf):
                continue
            if abs(cf) <= 1e-8:
                return cx
        
        # If we have a good candidate, try to refine it with local search
        if best_abs_f < 1.0:  # Only if we found something reasonably close
            # Local ternary search on |f(x)|^2 in a small neighborhood
            best_x_local = best_x
            best_f_local = best_abs_f
            h = 0.1
            if abs(x) > 1e3:
                h = 10.0
            elif abs(x) > 1:
                h = 1.0
            else:
                h = 0.01
            
            for i in range(20):
                if not time_check() or calls >= max_calls - 10:
                    break
                # Evaluate at best_x_local ± h
                x1 = best_x_local - h
                x2 = best_x_local + h
                f1 = _f(x1)
                f2 = _f(x2)
                
                if math.isnan(f1) and math.isnan(f2):
                    break
                
                if math.isnan(f1):
                    # Move right
                    best_x_local = best_x_local + h * 0.5
                    h *= 0.5
                    continue
                if math.isnan(f2):
                    # Move left
                    best_x_local = best_x_local - h * 0.5
                    h *= 0.5
                    continue
                
                af1 = abs(f1)
                af2 = abs(f2)
                
                if af1 <= 1e-8:
                    return x1
                if af2 <= 1e-8:
                    return x2
                
                if af1 < af2:
                    # Minimum is to the left
                    best_x_local = best_x_local - h * 0.3
                    h *= 0.7
                else:
                    # Minimum is to the right
                    best_x_local = best_x_local + h * 0.3
                    h *= 0.7
                
                if h < 1e-12:
                    break
        
        # Final check
        final_f = _f(best_x)
        if not math.isnan(final_f) and abs(final_f) <= 1e-8:
            return best_x

    return best_x
