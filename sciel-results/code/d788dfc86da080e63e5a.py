"""Root finding for ill-conditioned equations: multi-strategy robust solver."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Returns the best x found (minimizing |f(x)|) within the budget.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    max_evals = 500
    evals_used = 0
    best_x = float(x0)
    best_fx = None
    best_abs = float('inf')
    
    def check_time():
        return time.time() < deadline
    
    def update_best(x, fx):
        nonlocal best_x, best_fx, best_abs
        if fx is not None and math.isfinite(fx):
            afx = abs(fx)
            if afx < best_abs:
                best_x = x
                best_fx = fx
                best_abs = afx
            return afx
        return None
    
    def safe_eval(x):
        nonlocal evals_used
        if not check_time():
            return None
        if evals_used >= max_evals:
            return None
        if not math.isfinite(x):
            return None
        try:
            val = f(x)
            evals_used += 1
            if isinstance(val, (int, float)) and math.isfinite(val):
                return float(val)
            return None
        except Exception:
            evals_used += 1
            return None
    
    def estimate_derivative(x, fx):
        """Estimate derivative using Richardson extrapolation for better accuracy."""
        if evals_used >= max_evals - 4:
            return None
        
        # Adaptive base step
        base_h = 1e-7 * max(1.0, abs(x))
        
        # Try multiple step sizes for Richardson extrapolation
        results = []
        for h_mult in [1.0, 0.5, 0.25]:
            if evals_used >= max_evals - 2:
                break
            h = h_mult * base_h
            if h < 1e-14 * max(1.0, abs(x)):
                h = 1e-14 * max(1.0, abs(x))
            
            fp = safe_eval(x + h)
            if fp is None:
                continue
            fm = safe_eval(x - h)
            if fm is not None:
                df = (fp - fm) / (2.0 * h)
                results.append((h, df))
            else:
                df = (fp - fx) / h
                results.append((h, df))
        
        if not results:
            return None
        
        # Use Richardson extrapolation if we have multiple estimates
        if len(results) >= 2:
            # Sort by step size (smallest first)
            results.sort(key=lambda r: r[0])
            h1, df1 = results[0]
            h2, df2 = results[1]
            # Richardson extrapolation: assume error ~ h^2
            if abs(h1 - h2) > 1e-15:
                ratio = (h2 / h1) ** 2
                df_richardson = (ratio * df1 - df2) / (ratio - 1.0)
                if math.isfinite(df_richardson) and abs(df_richardson) > 1e-18:
                    return df_richardson
        
        # Fall back to the smallest step estimate
        h, df = results[0]
        if math.isfinite(df) and abs(df) > 1e-18:
            return df
        return None
    
    # Initial evaluation with retries at different points
    fx = safe_eval(best_x)
    if fx is not None:
        update_best(best_x, fx)
    else:
        for dx in [1e-10, -1e-10, 1e-6, -1e-6, 1e-3, -1e-3, 1.0, -1.0, 10.0, -10.0]:
            if not check_time() or evals_used >= max_evals:
                break
            trial_x = float(x0) + dx
            trial_fx = safe_eval(trial_x)
            if trial_fx is not None:
                afx = update_best(trial_x, trial_fx)
                if afx is not None:
                    best_x = trial_x
                    best_fx = trial_fx
                    break
        if best_fx is None:
            return best_x
        if best_abs <= 1e-8:
            return best_x
    
    if best_abs <= 1e-8:
        return best_x
    
    # Strategy 1: Improved Newton with adaptive damping and Richardson derivative
    x = best_x
    fx = best_fx
    
    for it in range(50):
        if not check_time() or evals_used >= max_evals - 15:
            break
        if abs(fx) <= 1e-8:
            return x
        
        df = estimate_derivative(x, fx)
        if df is None or abs(df) < 1e-18:
            break
        
        step = fx / df
        # Limit step to prevent divergence
        max_step = 100.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
        
        x_new = x - step
        if not math.isfinite(x_new):
            break
        
        fx_new = safe_eval(x_new)
        if fx_new is None:
            # Try damped steps
            for damp in [0.5, 0.25, 0.1]:
                x_new = x - damp * step
                fx_new = safe_eval(x_new)
                if fx_new is not None:
                    break
            if fx_new is None:
                break
        
        afx_new = abs(fx_new)
        update_best(x_new, fx_new)
        
        if afx_new <= 1e-8:
            return x_new
        
        # Adaptive damping: if function value increased, use smaller step
        if afx_new > abs(fx):
            # Try progressively smaller steps
            found = False
            for damp in [0.5, 0.25, 0.1, 0.05]:
                x_trial = x - damp * step
                fx_trial = safe_eval(x_trial)
                if fx_trial is None:
                    continue
                afx_trial = abs(fx_trial)
                update_best(x_trial, fx_trial)
                if afx_trial <= 1e-8:
                    return x_trial
                if afx_trial < abs(fx):
                    x_new = x_trial
                    fx_new = fx_trial
                    found = True
                    break
            
            if not found:
                # Keep original point, try secant next
                break
        
        # Check convergence
        if abs(step) < 1e-14 * max(1.0, abs(x_new)):
            break
        
        x = x_new
        fx = fx_new
    
    if best_abs <= 1e-8:
        return best_x
    
    # Strategy 2: Systematic bracket search with multiple scales
    a = None
    b = None
    fa = None
    fb = None
    
    center = best_x
    base = max(1.0, abs(center))
    
    # Search for sign change in multiple directions and scales
    scales = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
    
    for scale in scales:
        if not check_time() or evals_used >= max_evals - 30:
            break
        delta = scale * base
        if delta < 1e-14 * base:
            continue
            
        # Try multiple bracket configurations
        candidates = [
            (center - delta, center + delta),
            (center - 2*delta, center + 2*delta),
            (center, center + delta),
            (center - delta, center),
            (center - 3*delta, center + delta),
            (center - delta, center + 3*delta),
        ]
        
        for (xa, xb) in candidates:
            if not check_time() or evals_used >= max_evals - 20:
                break
            fa = safe_eval(xa)
            if fa is None:
                continue
            fb = safe_eval(xb)
            if fb is None:
                continue
            update_best(xa, fa)
            update_best(xb, fb)
            if best_abs <= 1e-8:
                return best_x
            if fa * fb < 0:
                a, b = xa, xb
                break
        if a is not None:
            break
    
    if a is not None and b is not None and check_time() and evals_used < max_evals - 10:
        # Bisection with early termination
        fa = safe_eval(a)
        fb = safe_eval(b)
        if fa is not None and fb is not None and fa * fb < 0:
            for _ in range(100):
                if not check_time() or evals_used >= max_evals - 3:
                    break
                mid = 0.5 * (a + b)
                fm = safe_eval(mid)
                if fm is None:
                    break
                update_best(mid, fm)
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm <= 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
                if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                    break
    
    if best_abs <= 1e-8:
        return best_x
    
    # Strategy 3: Secant method with restart
    if check_time() and evals_used < max_evals - 15:
        x0s = best_x
        fx0 = best_fx
        
        # Try multiple initial steps for secant
        for step_mult in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
            if not check_time() or evals_used >= max_evals - 10:
                break
            step_init = step_mult * max(1.0, abs(x0s))
            x1s = x0s + step_init
            fx1 = safe_eval(x1s)
            if fx1 is None:
                x1s = x0s - step_init
                fx1 = safe_eval(x1s)
            if fx1 is None:
                continue
            update_best(x1s, fx1)
            if best_abs <= 1e-8:
                return best_x
            
            # Run secant iterations
            for _ in range(30):
                if not check_time() or evals_used >= max_evals - 5:
                    break
                if abs(fx1) < abs(fx0):
                    x0s, x1s = x1s, x0s
                    fx0, fx1 = fx1, fx0
                if abs(fx1) <= 1e-8:
                    return x1s
                denom = fx1 - fx0
                if abs(denom) < 1e-18:
                    break
                x2s = x1s - fx1 * (x1s - x0s) / denom
                if not math.isfinite(x2s):
                    break
                max_step = 50.0 * max(1.0, abs(x1s))
                if abs(x2s - x1s) > max_step:
                    x2s = x1s - math.copysign(max_step, x2s - x1s)
                fx2 = safe_eval(x2s)
                if fx2 is None:
                    break
                update_best(x2s, fx2)
                if abs(fx2) <= 1e-8:
                    return x2s
                x0s, x1s = x1s, x2s
                fx0, fx1 = fx1, fx2
            
            if best_abs <= 1e-8:
                return best_x
    
    # Strategy 4: Local minimization polish (golden section on |f(x)|)
    if check_time() and evals_used < max_evals - 20 and best_abs > 1e-10:
        center = best_x
        base = max(1.0, abs(center))
        
        # Try to find a local minimum of |f(x)| using golden section
        for scale in [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]:
            if not check_time() or evals_used >= max_evals - 10:
                break
            delta = scale * base
            if delta < 1e-14 * base:
                continue
            
            # Golden section search on [center - delta, center + delta]
            a_g = center - delta
            b_g = center + delta
            fa_g = safe_eval(a_g)
            fb_g = safe_eval(b_g)
            if fa_g is None or fb_g is None:
                continue
            update_best(a_g, fa_g)
            update_best(b_g, fb_g)
            
            if best_abs <= 1e-8:
                return best_x
            
            # Golden ratio
            gr = (math.sqrt(5.0) - 1.0) / 2.0
            c = b_g - gr * (b_g - a_g)
            d = a_g + gr * (b_g - a_g)
            
            fc = safe_eval(c)
            fd = safe_eval(d)
            if fc is None or fd is None:
                continue
            update_best(c, fc)
            update_best(d, fd)
            
            if best_abs <= 1e-8:
                return best_x
            
            # Iterate golden section
            for _ in range(20):
                if not check_time() or evals_used >= max_evals - 5:
                    break
                if abs(fc) < abs(fd):
                    b_g = d
                    fb_g = fd
                    d = c
                    fd = fc
                    c = b_g - gr * (b_g - a_g)
                    fc = safe_eval(c)
                else:
                    a_g = c
                    fa_g = fc
                    c = d
                    fc = fd
                    d = a_g + gr * (b_g - a_g)
                    fd = safe_eval(d)
                
                if fc is None or fd is None:
                    break
                update_best(c, fc)
                update_best(d, fd)
                
                if best_abs <= 1e-8:
                    return best_x
                
                if abs(b_g - a_g) < 1e-14 * max(1.0, abs(center)):
                    break
    
    # Strategy 5: Fine grid search if still stuck
    if check_time() and evals_used < max_evals - 10 and best_abs > 1e-10:
        center = best_x
        base = max(1.0, abs(center))
        for power in range(-12, 3):
            if not check_time() or evals_used >= max_evals - 3:
                break
            delta = (10 ** power) * base
            if delta < 1e-15 * base:
                continue
            for sign in [1, -1]:
                if not check_time() or evals_used >= max_evals - 2:
                    break
                trial = center + sign * delta
                fval = safe_eval(trial)
                if fval is not None:
                    update_best(trial, fval)
                    if best_abs <= 1e-8:
                        return best_x
    
    return best_x
# EVOLVE-BLOCK-END
