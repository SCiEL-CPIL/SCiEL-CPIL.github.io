"""Root finding for ill-conditioned problems with evaluation budget awareness."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def time_up():
        return time.time() >= deadline
    
    x0 = float(x0)
    eval_count = 0
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)
    eval_count += 1
    
    def safe_eval(x):
        nonlocal eval_count
        if time_up():
            return None
        try:
            fx = f(x)
            eval_count += 1
            return fx
        except (ValueError, OverflowError, ZeroDivisionError):
            eval_count += 1
            return None
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if fx is not None:
            ax = abs(fx)
            if ax < best_abs:
                best_x = x
                best_f = fx
                best_abs = ax
            return ax
        return best_abs
    
    # If already good enough
    if best_abs <= 1e-8:
        return best_x
    
    # Phase 1: Inverse quadratic interpolation with bisection fallback
    # This uses global curve fitting to 3 points, avoiding finite-difference noise
    h0 = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h0
    f1 = safe_eval(x1)
    
    if f1 is None:
        x1 = x0 - h0
        f1 = safe_eval(x1)
    
    if f1 is not None:
        update_best(x1, f1)
        if best_abs <= 1e-8:
            return best_x
        
        # Get a third point for IQUO
        h1 = 1e-2 * max(1.0, abs(x0))
        x2 = x0 + h1
        f2 = safe_eval(x2)
        
        if f2 is None:
            x2 = x0 - h1
            f2 = safe_eval(x2)
        
        if f2 is not None:
            update_best(x2, f2)
            if best_abs <= 1e-8:
                return best_x
            
            # IQUO iteration with three points
            # Maintain three points: xa, xb, xc with corresponding fa, fb, fc
            # We'll track them and use inverse quadratic interpolation
            xa, xb, xc = x0, x1, x2
            fa, fb, fc = best_f, f1, f2
            
            for _ in range(100):
                if time_up():
                    break
                
                if abs(fb) <= 1e-12:
                    return xb
                if abs(fa) <= 1e-12:
                    return xa
                if abs(fc) <= 1e-12:
                    return xc
                
                # Inverse quadratic interpolation
                # Fit parabola x = A*f^2 + B*f + C through (fa,xa), (fb,xb), (fc,xc)
                # Then find x where f=0
                denom_a = (fa - fb) * (fa - fc)
                denom_b = (fb - fa) * (fb - fc)
                denom_c = (fc - fa) * (fc - fb)
                
                if abs(denom_a) < 1e-300 or abs(denom_b) < 1e-300 or abs(denom_c) < 1e-300:
                    # Degenerate case, fall back to secant
                    if abs(fa - fb) < 1e-300:
                        break
                    x_new = xb - fb * (xb - xa) / (fb - fa)
                else:
                    # Lagrange interpolation for x as function of f
                    x_new = (fa * fb * xa / denom_a + fb * fc * xb / denom_b + fc * fa * xc / denom_c)
                
                # Check if x_new is reasonable
                if not math.isfinite(x_new):
                    # Fall back to bisection-like step
                    mid = 0.5 * (xa + xc)
                    x_new = mid
                
                # Ensure x_new is within a reasonable range
                min_x = min(xa, xb, xc)
                max_x = max(xa, xb, xc)
                range_width = max_x - min_x
                if range_width > 0:
                    if x_new < min_x - 2 * range_width or x_new > max_x + 2 * range_width:
                        # Too far, use secant instead
                        if abs(fa - fb) < 1e-300:
                            break
                        x_new = xb - fb * (xb - xa) / (fb - fa)
                
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                
                abs_new = update_best(x_new, f_new)
                if abs_new <= 1e-12:
                    return x_new
                
                # Update the three points: keep the two with smallest |f| and the new one
                # Sort by |f|
                points = [(abs(fa), xa, fa), (abs(fb), xb, fb), (abs(fc), xc, fc), (abs(f_new), x_new, f_new)]
                points.sort(key=lambda p: p[0])
                # Keep the three best
                (abs_fa, xa, fa), (abs_fb, xb, fb), (abs_fc, xc, fc) = points[0], points[1], points[2]
                
                # Check for stagnation
                if abs(x_new - xb) < 1e-14 * max(1.0, abs(xb)):
                    break
        
        # If IQUO didn't converge, try secant as fallback
        if best_abs > 1e-8 and not time_up():
            # Use the two best points for secant
            # Re-evaluate to get current best
            x_prev = best_x
            f_prev = best_f
            
            # Get a second point
            h_sec = 1e-6 * max(1.0, abs(best_x))
            x_curr = best_x + h_sec
            f_curr = safe_eval(x_curr)
            if f_curr is None:
                x_curr = best_x - h_sec
                f_curr = safe_eval(x_curr)
            
            if f_curr is not None:
                update_best(x_curr, f_curr)
                if best_abs <= 1e-8:
                    return best_x
                
                for _ in range(80):
                    if time_up():
                        break
                    if abs(f_curr) <= 1e-12:
                        return x_curr
                    if abs(f_prev) <= 1e-12:
                        return x_prev
                    
                    denom = f_curr - f_prev
                    if abs(denom) < 1e-300:
                        break
                    
                    x_new = x_curr - f_curr * (x_curr - x_prev) / denom
                    f_new = safe_eval(x_new)
                    
                    if f_new is None:
                        break
                    
                    abs_new = update_best(x_new, f_new)
                    if abs_new <= 1e-12:
                        return x_new
                    
                    if abs(x_new - x_curr) < 1e-14 * max(1.0, abs(x_curr)):
                        break
                    
                    x_prev = x_curr
                    f_prev = f_curr
                    x_curr = x_new
                    f_curr = f_new
    
    # Phase 2: Bisection safety net
    if not time_up() and best_abs > 1e-8:
        step = 1e-3 * max(1.0, abs(best_x))
        for trial in range(30):
            if time_up():
                break
            x_left = best_x - step * (trial + 1)
            x_right = best_x + step * (trial + 1)
            f_left = safe_eval(x_left)
            f_right = safe_eval(x_right)
            
            if f_left is None or f_right is None:
                break
            
            update_best(x_left, f_left)
            update_best(x_right, f_right)
            
            if best_abs <= 1e-8:
                return best_x
            
            if f_left * f_right < 0:
                a, b = x_left, x_right
                fa, fb = f_left, f_right
                for _ in range(80):
                    if time_up():
                        break
                    m = 0.5 * (a + b)
                    fm = safe_eval(m)
                    if fm is None:
                        break
                    abs_m = update_best(m, fm)
                    if abs_m <= 1e-12:
                        return m
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    if abs(b - a) < 1e-14 * max(1.0, abs(m)):
                        break
                break
            elif f_left * best_f < 0:
                a, b = x_left, best_x
                fa, fb = f_left, best_f
                for _ in range(80):
                    if time_up():
                        break
                    m = 0.5 * (a + b)
                    fm = safe_eval(m)
                    if fm is None:
                        break
                    abs_m = update_best(m, fm)
                    if abs_m <= 1e-12:
                        return m
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    if abs(b - a) < 1e-14 * max(1.0, abs(m)):
                        break
                break
            elif best_f * f_right < 0:
                a, b = best_x, x_right
                fa, fb = best_f, f_right
                for _ in range(80):
                    if time_up():
                        break
                    m = 0.5 * (a + b)
                    fm = safe_eval(m)
                    if fm is None:
                        break
                    abs_m = update_best(m, fm)
                    if abs_m <= 1e-12:
                        return m
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    if abs(b - a) < 1e-14 * max(1.0, abs(m)):
                        break
                break
            else:
                step *= 2
                continue
    
    return best_x
# EVOLVE-BLOCK-END
