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

    def time_ok():
        return time.time() - start_time < TIME_LIMIT - 5.0

    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Global multi-scale grid search to find sign-change bracket
    # Evaluate f at exponentially spaced points in both directions
    a, b, fa, fb = None, None, None, None
    found_bracket = False
    
    # Determine base scale from x0
    if abs(x) > 1e6:
        base = 1e6
    elif abs(x) > 1e3:
        base = 1e3
    elif abs(x) > 1:
        base = 1.0
    else:
        base = 0.1
    
    # Try multiple scales: from fine to coarse
    # For each scale, evaluate at points x ± k*scale for k = 1, 2, 4, 8, ...
    # Look for sign change between consecutive evaluated points
    
    # First pass: fine-grained search around x0
    scales = []
    for k in range(20):
        scales.append(base * (0.1 ** k))  # very fine to fine
        scales.append(base * (10 ** k))   # fine to very coarse
    # Remove duplicates and sort by magnitude
    scales = sorted(set(scales), key=abs)
    
    # Evaluate at points x ± s for each scale, tracking previous point
    prev_x, prev_fx = None, None
    found_points = []  # list of (x, fx) in evaluation order
    
    # Evaluate at x0 first (already have fx)
    found_points.append((x, fx))
    
    # For each scale, try positive and negative offsets
    for s in scales:
        if not time_ok() or calls >= max_calls - 30:
            break
        
        # Positive direction
        xp = x + s
        fp = _f(xp)
        if not math.isnan(fp):
            found_points.append((xp, fp))
            if abs(fp) <= 1e-8:
                return xp
            # Check for sign change with any recent point
            for (px, pfx) in found_points[-5:]:
                if px != xp and pfx * fp < 0:
                    a, b, fa, fb = px, xp, pfx, fp
                    if a > b:
                        a, b = b, a
                        fa, fb = fb, fa
                    found_bracket = True
                    break
            if found_bracket:
                break
        
        # Negative direction
        xn = x - s
        fn = _f(xn)
        if not math.isnan(fn):
            found_points.append((xn, fn))
            if abs(fn) <= 1e-8:
                return xn
            # Check for sign change with any recent point
            for (px, pfx) in found_points[-5:]:
                if px != xn and pfx * fn < 0:
                    a, b, fa, fb = px, xn, pfx, fn
                    if a > b:
                        a, b = b, a
                        fa, fb = fb, fa
                    found_bracket = True
                    break
            if found_bracket:
                break
    
    # If no bracket found yet, do a denser search using all collected points
    if not found_bracket and len(found_points) > 1:
        # Sort by x and look for sign changes
        sorted_pts = sorted(found_points, key=lambda p: p[0])
        for i in range(len(sorted_pts) - 1):
            (x1, f1), (x2, f2) = sorted_pts[i], sorted_pts[i+1]
            if f1 * f2 < 0:
                a, b, fa, fb = x1, x2, f1, f2
                found_bracket = True
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
    
    # If still no bracket, try adaptive expansion in both directions
    if not found_bracket:
        # Expand outward from x0 with increasing steps
        step = base * 0.5
        last_x, last_fx = x, fx
        for direction in [1, -1]:
            if not time_ok() or calls >= max_calls - 50:
                break
            cur_x = x + direction * step
            for i in range(25):
                if not time_ok() or calls >= max_calls - 50:
                    break
                cf = _f(cur_x)
                if math.isnan(cf):
                    break
                if abs(cf) <= 1e-8:
                    return cur_x
                if last_fx * cf < 0:
                    a, b, fa, fb = last_x, cur_x, last_fx, cf
                    if a > b:
                        a, b = b, a
                        fa, fb = fb, fa
                    found_bracket = True
                    break
                last_x, last_fx = cur_x, cf
                step *= 1.8
                cur_x = x + direction * step
            if found_bracket:
                break
    
    if found_bracket:
        # Illinois method for efficient bisection-like convergence
        for i in range(120):
            if not time_ok() or calls >= max_calls - 10:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            
            # Illinois: use secant if it stays in bracket, else bisection
            if abs(fa - fb) > 1e-30:
                m = (a * fb - b * fa) / (fb - fa)
            else:
                m = (a + b) / 2
            
            if a < m < b:
                fm = _f(m)
                if math.isnan(fm):
                    break
                if abs(fm) <= 1e-8:
                    return m
                if fa * fm < 0:
                    b, fb = m, fm
                    fa *= 0.5  # Illinois damping
                else:
                    a, fa = m, fm
                    fb *= 0.5
            else:
                m = (a + b) / 2
                fm = _f(m)
                if math.isnan(fm):
                    break
                if abs(fm) <= 1e-8:
                    return m
                if fa * fm < 0:
                    b, fb = m, fm
                    fa *= 0.5
                else:
                    a, fa = m, fm
                    fb *= 0.5
            
            # Convergence check
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
        
        # Newton polish from the better endpoint
        result_x = a if abs(fa) < abs(fb) else b
        if time_ok() and calls < max_calls - 5:
            nx = result_x
            for _ in range(12):
                if not time_ok() or calls >= max_calls - 2:
                    break
                fn = _f(nx)
                if math.isnan(fn):
                    break
                if abs(fn) <= 1e-8:
                    return nx
                h = max(1e-8 * abs(nx), 1e-12)
                fn_plus = _f(nx + h)
                if math.isnan(fn_plus):
                    break
                df = (fn_plus - fn) / h
                if abs(df) > 1e-30:
                    new_x = nx - fn / df
                    if math.isfinite(new_x):
                        # Damped Newton: don't overshoot bracket too far
                        if a is not None and b is not None:
                            lo = min(a, b)
                            hi = max(a, b)
                            if new_x < lo - 2*(hi-lo) or new_x > hi + 2*(hi-lo):
                                new_x = (lo + hi) / 2
                        nx = new_x
                    else:
                        break
                else:
                    break
            if time_ok() and calls < max_calls:
                f_final = _f(nx)
                if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                    return nx
                if not math.isnan(f_final) and abs(f_final) < abs(_f(result_x)):
                    return nx
        
        return result_x
    
    # No bracket found: use best |f(x)| point and do local search
    if time_ok() and calls < max_calls - 30:
        # Local minimization of |f(x)| using golden section search
        best_x_local = best_x
        best_f_local = best_abs_f
        
        # Determine initial search window
        h = 0.1
        if abs(x) > 1e3:
            h = 10.0
        elif abs(x) > 1:
            h = 1.0
        else:
            h = 0.01
        
        # Golden section search on |f|^2
        gr = (1 + math.sqrt(5)) / 2  # golden ratio
        lo = best_x_local - h
        hi = best_x_local + h
        
        for i in range(30):
            if not time_ok() or calls >= max_calls - 10:
                break
            
            m1 = hi - (hi - lo) / gr
            m2 = lo + (hi - lo) / gr
            
            f1 = _f(m1)
            f2 = _f(m2)
            
            if math.isnan(f1) and math.isnan(f2):
                break
            if math.isnan(f1):
                hi = m2
                continue
            if math.isnan(f2):
                lo = m1
                continue
            
            af1 = abs(f1)
            af2 = abs(f2)
            
            if af1 <= 1e-8:
                return m1
            if af2 <= 1e-8:
                return m2
            
            if af1 < af2:
                hi = m2
                if af1 < best_f_local:
                    best_f_local = af1
                    best_x_local = m1
            else:
                lo = m1
                if af2 < best_f_local:
                    best_f_local = af2
                    best_x_local = m2
            
            if abs(hi - lo) < 1e-12 * max(1.0, abs(lo), abs(hi)):
                break
        
        # Newton polish
        if time_ok() and calls < max_calls - 5:
            nx = best_x_local
            for _ in range(10):
                if not time_ok() or calls >= max_calls - 2:
                    break
                fn = _f(nx)
                if math.isnan(fn):
                    break
                if abs(fn) <= 1e-8:
                    return nx
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
            if time_ok() and calls < max_calls:
                f_final = _f(nx)
                if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                    return nx
                if not math.isnan(f_final) and abs(f_final) < best_abs_f:
                    best_x = nx
                    best_abs_f = abs(f_final)
    
    return best_x
