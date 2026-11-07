import math
import time

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin for 120s limit
    max_calls = 495
    
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

    # 1. Global multi-scale grid search to find a bracket
    a, b, fa, fb = None, None, None, None
    found_bracket = False
    
    # Determine search domain based on x0
    if abs(x) > 1e6:
        domain_scale = 1e6
    elif abs(x) > 1e3:
        domain_scale = 1e3
    elif abs(x) > 1:
        domain_scale = 1.0
    else:
        domain_scale = 0.1
    
    # Multi-scale grid: exponentially spaced points
    # Start with a coarse grid, then refine
    grid_points = []
    base_step = domain_scale * 0.1
    
    # Generate points at exponentially increasing distances from x0
    for power in range(1, 10):
        step = base_step * (1.5 ** power)
        grid_points.append(x + step)
        grid_points.append(x - step)
    
    # Sort by distance from x0 to prioritize nearby points
    grid_points.sort(key=lambda p: abs(p - x))
    
    # Evaluate grid points to find sign changes
    prev_x, prev_f = x, fx
    for gp in grid_points:
        if not time_check() or calls >= max_calls - 80:
            break
        gf = _f(gp)
        if math.isnan(gf):
            continue
        if abs(gf) <= 1e-8:
            return gp
        if prev_f * gf < 0:
            a, b, fa, fb = prev_x, gp, prev_f, gf
            found_bracket = True
            break
        prev_x, prev_f = gp, gf
    
    # If no bracket found in grid, try local expansion
    if not found_bracket:
        step = base_step
        for i in range(1, 20):
            if not time_check() or calls >= max_calls - 60:
                break
            xp = x + step
            fp = _f(xp)
            if math.isnan(fp):
                step *= 1.5
                continue
            if abs(fp) <= 1e-8:
                return xp
            if fx * fp < 0:
                a, b, fa, fb = x, xp, fx, fp
                found_bracket = True
                break
            step *= 1.5
        
        if not found_bracket:
            step = base_step
            for i in range(1, 20):
                if not time_check() or calls >= max_calls - 60:
                    break
                xn = x - step
                fn = _f(xn)
                if math.isnan(fn):
                    step *= 1.5
                    continue
                if abs(fn) <= 1e-8:
                    return xn
                if fx * fn < 0:
                    a, b, fa, fb = xn, x, fn, fx
                    found_bracket = True
                    break
                step *= 1.5

    if found_bracket:
        # 2. Refine using Illinois method (converges faster than bisection)
        for i in range(100):
            if not time_check() or calls >= max_calls - 15:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            
            # Illinois method: weighted midpoint
            if abs(fa) < abs(fb):
                m = (a + b) / 2
                fm = _f(m)
                if math.isnan(fm):
                    break
                if abs(fm) <= 1e-8:
                    return m
                if fa * fm < 0:
                    b, fb = m, fm
                    a, fa = 2 * a - b, fa  # Weight toward a
                else:
                    a, fa = m, fm
                    b, fb = 2 * b - a, fb  # Weight toward b
            else:
                m = (a + b) / 2
                fm = _f(m)
                if math.isnan(fm):
                    break
                if abs(fm) <= 1e-8:
                    return m
                if fa * fm < 0:
                    b, fb = m, fm
                    b, fb = 2 * b - a, fb  # Weight toward b
                else:
                    a, fa = m, fm
                    a, fa = 2 * a - b, fa  # Weight toward a
        
        # Final high-precision polish: try Newton's method with finite difference
        result_x = a if abs(fa) < abs(fb) else b
        if time_check() and calls < max_calls - 5:
            # Try a few Newton steps from the best point
            nx = result_x
            for _ in range(10):
                if not time_check() or calls >= max_calls - 2:
                    break
                fn = _f(nx)
                if math.isnan(fn):
                    break
                if abs(fn) <= 1e-8:
                    return nx
                # Finite difference derivative
                h = max(1e-8 * abs(nx), 1e-12)
                fn_plus = _f(nx + h)
                if math.isnan(fn_plus):
                    break
                df = (fn_plus - fn) / h
                if abs(df) > 1e-30:
                    new_x = nx - fn / df
                    if math.isfinite(new_x):
                        nx = new_x
                    else:
                        break
                else:
                    break
            # Final check on polished result
            if time_check() and calls < max_calls:
                f_final = _f(nx)
                if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                    return nx
                # If Newton didn't help, return the better of original and polished
                if not math.isnan(f_final) and abs(f_final) < abs(_f(result_x)):
                    return nx
        
        return result_x

    # 3. If no bracket found, do local minimization of |f(x)|
    # Use a simple expanding search for minimum |f|
    if time_check() and calls < max_calls - 30:
        # Try a few candidate points around x0
        candidates = [x]
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
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
            
            for i in range(25):
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
            
            # Final Newton polish on the local minimum
            if time_check() and calls < max_calls - 5:
                nx = best_x_local
                for _ in range(10):
                    if not time_check() or calls >= max_calls - 2:
                        break
                    fn = _f(nx)
                    if math.isnan(fn):
                        break
                    if abs(fn) <= 1e-8:
                        return nx
                    # Finite difference derivative
                    h = max(1e-8 * abs(nx), 1e-12)
                    fn_plus = _f(nx + h)
                    if math.isnan(fn_plus):
                        break
                    df = (fn_plus - fn) / h
                    if abs(df) > 1e-30:
                        new_x = nx - fn / df
                        if math.isfinite(new_x):
                            nx = new_x
                        else:
                            break
                    else:
                        break
                # Final check
                if time_check() and calls < max_calls:
                    f_final = _f(nx)
                    if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                        return nx
                    if not math.isnan(f_final) and abs(f_final) < best_abs_f:
                        best_x = nx
                        best_abs_f = abs(f_final)
        
        # Final check
        final_f = _f(best_x)
        if not math.isnan(final_f) and abs(final_f) <= 1e-8:
            return best_x

    return best_x
