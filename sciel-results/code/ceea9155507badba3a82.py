"""Root finding for ill-conditioned problems with robust hybrid strategy."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid strategy: Secant method with safeguards, bisection fallback,
    and a final high-precision polish pass for ill-conditioned cases.
    """
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    eval_count = 0
    max_evals = 500
    best_x = x0
    best_fx = f(x0)
    best_abs_fx = abs(best_fx)
    eval_count += 1
    
    if best_abs_fx <= 1e-8:
        return best_x
    
    def eval_f(x):
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if eval_count >= max_evals:
            return best_fx  # Return last known value as fallback
        fx = f(x)
        eval_count += 1
        if not math.isfinite(fx):
            fx = 1e300 * (1 if fx > 0 else -1)
        if abs(fx) < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs(fx)
        return fx
    
    def time_exceeded():
        return time.time() >= deadline
    
    # Helper: Secant step using two previous points (saves 1 eval vs Newton)
    def secant_step(x1, f1, x0_prev, f0_prev):
        if time_exceeded() or eval_count >= max_evals:
            return None
        if abs(x1 - x0_prev) < 1e-15 * max(1.0, abs(x1)):
            # Points too close, use numerical derivative
            h = 1e-7 * max(1.0, abs(x1))
            f_xh = eval_f(x1 + h)
            if eval_count >= max_evals:
                return None
            d = (f_xh - f1) / h
        else:
            d = (f1 - f0_prev) / (x1 - x0_prev)
        if d == 0 or not math.isfinite(d):
            return None
        step = f1 / d
        # Limit step size to avoid divergence
        max_step = 1.0 * max(1.0, abs(x1))
        if abs(step) > max_step:
            step = max_step * (1 if step > 0 else -1)
        return x1 + step
    
    # Helper: Try to find a bracket around x
    def find_bracket(x, fx, max_steps=20):
        if time_exceeded() or eval_count >= max_evals - 10:
            return None
        # Try expanding in both directions
        for direction in [1, -1]:
            a = x
            fa = fx
            step = 1e-3 * max(1.0, abs(x))
            for _ in range(max_steps):
                if time_exceeded() or eval_count >= max_evals - 5:
                    break
                b = a + direction * step
                fb = eval_f(b)
                if eval_count >= max_evals:
                    break
                if fa * fb <= 0:
                    return (a, b, fa, fb)
                a = b
                fa = fb
                step *= 2.0
                if step > 1e6:
                    break
        return None
    
    # Helper: Bisection within [a, b]
    def bisection(a, b, fa, fb, max_iter=50):
        nonlocal best_x, best_fx, best_abs_fx
        for _ in range(max_iter):
            if time_exceeded() or eval_count >= max_evals - 2:
                break
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                break
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            if eval_count >= max_evals:
                break
            if abs(fm) < best_abs_fx:
                best_x = mid
                best_fx = fm
                best_abs_fx = abs(fm)
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        return (a + b) / 2.0
    
    # Helper: Illinois method (improved regula falsi)
    def illinois(a, b, fa, fb, max_iter=50):
        nonlocal best_x, best_fx, best_abs_fx
        for _ in range(max_iter):
            if time_exceeded() or eval_count >= max_evals - 2:
                break
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                break
            # Regula falsi step
            if abs(fa) < abs(fb):
                c = b - fb * (b - a) / (fb - fa)
            else:
                c = a - fa * (b - a) / (fb - fa)
            fc = eval_f(c)
            if eval_count >= max_evals:
                break
            if abs(fc) < best_abs_fx:
                best_x = c
                best_fx = fc
                best_abs_fx = abs(fc)
            if abs(fc) <= 1e-12:
                return c
            if fa * fc <= 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                # Illinois modification: halve the larger function value
                if abs(fa) > abs(fb):
                    fa /= 2.0
        return (a + b) / 2.0
    
    # Helper: High-precision polish using secant method with tight tolerance
    def polish(x_start, fx_start, max_iter=20):
        """Final polish pass: use secant method to drive |f| below 1e-8."""
        nonlocal best_x, best_fx, best_abs_fx
        if time_exceeded() or eval_count >= max_evals - 3:
            return x_start, fx_start
        
        x1 = x_start
        f1 = fx_start
        
        # Get a second point for secant
        h = 1e-6 * max(1.0, abs(x1))
        x0_p = x1 - h
        f0_p = eval_f(x0_p)
        if eval_count >= max_evals:
            return x1, f1
        
        for _ in range(max_iter):
            if time_exceeded() or eval_count >= max_evals - 1:
                break
            if abs(f1) <= 1e-10:  # Tighter than required 1e-8
                return x1, f1
            
            # Secant step
            if abs(x1 - x0_p) < 1e-15 * max(1.0, abs(x1)):
                # Too close, use numerical derivative
                h2 = 1e-8 * max(1.0, abs(x1))
                f_xh = eval_f(x1 + h2)
                if eval_count >= max_evals:
                    break
                d = (f_xh - f1) / h2
            else:
                d = (f1 - f0_p) / (x1 - x0_p)
            
            if d == 0 or not math.isfinite(d):
                break
            
            step = f1 / d
            # Limit step
            max_step = 0.5 * max(1.0, abs(x1))
            if abs(step) > max_step:
                step = max_step * (1 if step > 0 else -1)
            
            x_new = x1 + step
            if not math.isfinite(x_new):
                break
            
            f_new = eval_f(x_new)
            if eval_count >= max_evals:
                break
            
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)
            
            if abs(f_new) <= 1e-10:
                return x_new, f_new
            
            # Update secant points: keep the better point
            if abs(f_new) < abs(f1):
                x0_p, f0_p = x1, f1
                x1, f1 = x_new, f_new
            else:
                x0_p, f0_p = x_new, f_new
        
        return x1, f1
    
    # Main algorithm: try multiple strategies
    # Strategy 1: Secant method with safeguards
    x = x0
    fx = best_fx
    x_prev = x0 - 1e-6 * max(1.0, abs(x0))
    fx_prev = eval_f(x_prev)
    if eval_count >= max_evals:
        return best_x
    
    if abs(fx_prev) < best_abs_fx:
        best_x = x_prev
        best_fx = fx_prev
        best_abs_fx = abs(fx_prev)
    
    secant_success = False
    for _ in range(40):
        if time_exceeded() or eval_count >= max_evals - 5:
            break
        if abs(fx) <= 1e-10:
            secant_success = True
            break
        
        x_new = secant_step(x, fx, x_prev, fx_prev)
        if x_new is None:
            break
        
        if not math.isfinite(x_new):
            break
        
        fx_new = eval_f(x_new)
        if eval_count >= max_evals:
            break
        
        # Accept step if it reduces |f| or is a small step
        if abs(fx_new) < abs(fx) or abs(x_new - x) < 1e-10:
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
        else:
            # Try damped secant
            x_half = x + (x_new - x) / 2.0
            fx_half = eval_f(x_half)
            if eval_count >= max_evals:
                break
            if abs(fx_half) < abs(fx):
                x_prev, fx_prev = x, fx
                x, fx = x_half, fx_half
            else:
                break
    
    if not secant_success and abs(fx) > 1e-8:
        # Strategy 2: Try to find a bracket and use bisection/Illinois
        bracket = find_bracket(x, fx)
        if bracket is not None:
            a, b, fa, fb = bracket
            # Use Illinois method for faster convergence
            result = illinois(a, b, fa, fb)
            fr = eval_f(result)
            if eval_count >= max_evals:
                return best_x
            if abs(fr) < best_abs_fx:
                best_x = result
                best_fx = fr
                best_abs_fx = abs(fr)
    
    # Strategy 3: If still not good enough, try restarts at different points
    if best_abs_fx > 1e-8 and eval_count < max_evals - 20 and not time_exceeded():
        restart_points = [
            x0 * 1.1,
            x0 * 0.9,
            x0 + 1.0,
            x0 - 1.0,
            x0 * 2.0,
            x0 / 2.0 if x0 != 0 else 1.0,
        ]
        for rp in restart_points:
            if time_exceeded() or eval_count >= max_evals - 15:
                break
            if abs(rp - x0) < 1e-12:
                continue
            
            # Quick secant from restart point
            xr = rp
            fxr = eval_f(xr)
            if eval_count >= max_evals:
                break
            if abs(fxr) < best_abs_fx:
                best_x = xr
                best_fx = fxr
                best_abs_fx = abs(fxr)
            if abs(fxr) <= 1e-8:
                return xr
            
            # Get second point for secant
            h_r = 1e-6 * max(1.0, abs(xr))
            xr_prev = xr - h_r
            fxr_prev = eval_f(xr_prev)
            if eval_count >= max_evals:
                break
            if abs(fxr_prev) < best_abs_fx:
                best_x = xr_prev
                best_fx = fxr_prev
                best_abs_fx = abs(fxr_prev)
            
            for _ in range(15):
                if time_exceeded() or eval_count >= max_evals - 5:
                    break
                if abs(fxr) <= 1e-10:
                    break
                
                x_new = secant_step(xr, fxr, xr_prev, fxr_prev)
                if x_new is None:
                    break
                if not math.isfinite(x_new):
                    break
                
                fx_new = eval_f(x_new)
                if eval_count >= max_evals:
                    break
                
                if abs(fx_new) < abs(fxr):
                    xr_prev, fxr_prev = xr, fxr
                    xr, fxr = x_new, fx_new
                    if abs(fxr) < best_abs_fx:
                        best_x = xr
                        best_fx = fxr
                        best_abs_fx = abs(fxr)
                    if abs(fxr) <= 1e-8:
                        return xr
                else:
                    break
    
    # Final high-precision polish pass
    if best_abs_fx > 1e-10 and eval_count < max_evals - 3 and not time_exceeded():
        x_polished, fx_polished = polish(best_x, best_fx)
        if abs(fx_polished) < best_abs_fx:
            best_x = x_polished
            best_fx = fx_polished
            best_abs_fx = abs(fx_polished)
    
    # Final check
    if best_abs_fx <= 1e-8:
        return best_x
    
    # Return best result found
    return best_x
# EVOLVE-BLOCK-END
