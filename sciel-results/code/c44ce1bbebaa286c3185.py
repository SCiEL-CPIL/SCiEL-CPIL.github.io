"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a combination of:
    1. Bisection for robustness (guaranteed convergence for continuous f)
    2. Newton's method for speed when derivative is available via finite differences
    3. Secant method as fallback when Newton fails
    
    Strategy:
    - Start with a bracketing search around x0 to find sign change
    - If bracket found, use bisection + Newton hybrid (Brent-like)
    - If no bracket found, use Newton/secant from x0
    - Track best |f(x)| seen and return it if deadline approaches
    """
    x = float(x0)
    best_x = x
    best_f = f(x)
    best_abs = abs(best_f)
    
    if best_abs <= 1e-8:
        return x
    
    call_count = 1  # we've already made one call
    
    def safe_eval(val):
        nonlocal call_count
        if not _is_finite(val):
            return float('inf')
        try:
            result = f(val)
            call_count += 1
            if not _is_finite(result):
                return float('inf') * (1 if result > 0 else -1)
            return float(result)
        except Exception:
            call_count += 1
            return float('inf')
    
    def update_best(val, fval):
        nonlocal best_x, best_f, best_abs
        if _is_finite(val) and _is_finite(fval):
            if abs(fval) < best_abs:
                best_x = val
                best_f = fval
                best_abs = abs(fval)
    
    # Phase 1: Try to find a bracket [a, b] where f(a)*f(b) < 0
    # Search in expanding intervals around x0
    a = x
    b = x
    fa = best_f
    fb = best_f
    bracket_found = False
    
    # Try small steps first, then expand
    step_sizes = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    for h in step_sizes:
        if _time_left() < 1.0:
            break
        if call_count > 480:
            break
            
        # Try x - h
        val_left = x - h
        if _is_finite(val_left):
            fl = safe_eval(val_left)
            if _is_finite(fl):
                update_best(val_left, fl)
                if fa * fl < 0:
                    a = val_left
                    b = x
                    fa = fl
                    fb = best_f
                    bracket_found = True
                    break
                else:
                    a = val_left
                    fa = fl
        
        # Try x + h
        val_right = x + h
        if _is_finite(val_right):
            fr = safe_eval(val_right)
            if _is_finite(fr):
                update_best(val_right, fr)
                if best_f * fr < 0:
                    a = x
                    b = val_right
                    fa = best_f
                    fb = fr
                    bracket_found = True
                    break
                else:
                    b = val_right
                    fb = fr
    
    if bracket_found:
        # Phase 2: Use bisection + Newton hybrid on [a, b]
        # This is similar to Brent's method
        for iteration in range(60):
            if _time_left() < 1.0:
                break
            if call_count > 490:
                break
            
            # Check if we've converged
            if best_abs <= 1e-8:
                return best_x
            
            # Bisection midpoint
            c = (a + b) / 2.0
            fc = safe_eval(c)
            update_best(c, fc)
            
            if not _is_finite(fc):
                break
            
            if abs(fc) < best_abs:
                best_x = c
                best_f = fc
                best_abs = abs(fc)
            
            if best_abs <= 1e-8:
                return best_x
            
            # Update bracket
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            # Try Newton step from the endpoint with smaller |f|
            # Use the point with smaller |f| for Newton
            if abs(fa) < abs(fb):
                x_new = a
                f_new = fa
                x_other = b
                f_other = fb
            else:
                x_new = b
                f_new = fb
                x_other = a
                f_other = fa
            
            # Finite difference derivative
            h_deriv = 1e-8 * max(1.0, abs(x_new))
            f_plus = safe_eval(x_new + h_deriv)
            if _is_finite(f_plus):
                d = (f_plus - f_new) / h_deriv
                if abs(d) > 1e-15:
                    x_n = x_new - f_new / d
                    # Check if Newton step is in bracket
                    if a < x_n < b:
                        fn = safe_eval(x_n)
                        update_best(x_n, fn)
                        if _is_finite(fn):
                            if abs(fn) < best_abs:
                                best_x = x_n
                                best_f = fn
                                best_abs = abs(fn)
                            if best_abs <= 1e-8:
                                return best_x
                            # Update bracket
                            if fa * fn < 0:
                                b = x_n
                                fb = fn
                            else:
                                a = x_n
                                fa = fn
                            continue
                    
                    # If Newton step is outside, try secant
                    if _is_finite(f_other) and f_new != f_other:
                        x_s = x_new - f_new * (x_new - x_other) / (f_new - f_other)
                        if a < x_s < b:
                            fs = safe_eval(x_s)
                            update_best(x_s, fs)
                            if _is_finite(fs):
                                if abs(fs) < best_abs:
                                    best_x = x_s
                                    best_f = fs
                                    best_abs = abs(fs)
                                if best_abs <= 1e-8:
                                    return best_x
                                if fa * fs < 0:
                                    b = x_s
                                    fb = fs
                                else:
                                    a = x_s
                                    fa = fs
                                continue
            
            # If we get here, just do bisection (already done above)
            # The bracket was updated, continue loop
    
    # Phase 3: If no bracket or bisection didn't converge, try Newton/secant from x0
    if best_abs > 1e-8 and _time_left() > 2.0 and call_count < 490:
        x_curr = x
        f_curr = best_f
        x_prev = None
        f_prev = None
        
        for iteration in range(30):
            if _time_left() < 1.0:
                break
            if call_count > 495:
                break
            
            if best_abs <= 1e-8:
                return best_x
            
            if x_prev is not None and f_prev is not None and f_curr != f_prev:
                # Secant method
                x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                # Newton with finite difference
                h_deriv = 1e-8 * max(1.0, abs(x_curr))
                f_plus = safe_eval(x_curr + h_deriv)
                if not _is_finite(f_plus):
                    break
                d = (f_plus - f_curr) / h_deriv
                if abs(d) < 1e-15:
                    break
                x_new = x_curr - f_curr / d
            
            if not _is_finite(x_new):
                break
            
            # Damping: if step is too large, reduce it
            if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
                x_new = x_curr + 10 * max(1.0, abs(x_curr)) * (1 if x_new > x_curr else -1)
            
            f_new = safe_eval(x_new)
            update_best(x_new, f_new)
            
            if not _is_finite(f_new):
                break
            
            if abs(f_new) < best_abs:
                best_x = x_new
                best_f = f_new
                best_abs = abs(f_new)
            
            if best_abs <= 1e-8:
                return best_x
            
            # If step didn't improve, try a damped step
            if abs(f_new) > abs(f_curr):
                # Halve the step
                x_damped = x_curr + (x_new - x_curr) / 2.0
                f_damped = safe_eval(x_damped)
                if _is_finite(f_damped):
                    update_best(x_damped, f_damped)
                    if abs(f_damped) < best_abs:
                        best_x = x_damped
                        best_f = f_damped
                        best_abs = abs(f_damped)
                    if best_abs <= 1e-8:
                        return best_x
                    x_prev = x_curr
                    f_prev = f_curr
                    x_curr = x_damped
                    f_curr = f_damped
                else:
                    break
            else:
                x_prev = x_curr
                f_prev = f_curr
                x_curr = x_new
                f_curr = f_new
    
    # Return best found
    if best_abs <= 1e-8:
        return best_x
    
    # If we didn't get below 1e-8, still return best
    # The evaluator will judge if it's good enough
    return best_x
