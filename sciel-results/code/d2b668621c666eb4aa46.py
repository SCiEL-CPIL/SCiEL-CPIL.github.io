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
    deadline = start_time + 118.0
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
    
    # Strategy 1: Adaptive step-size Newton with central differences
    x = best_x
    fx = best_fx
    newton_ok = False
    
    for it in range(40):
        if not check_time() or evals_used >= max_evals - 10:
            break
        if abs(fx) <= 1e-8:
            return x
        
        # Adaptive h for finite difference
        h = 1e-8 * max(1.0, abs(x))
        # Try to get derivative via central difference
        fp = safe_eval(x + h)
        if fp is None:
            break
        fm = safe_eval(x - h)
        if fm is not None:
            df = (fp - fm) / (2.0 * h)
        else:
            df = (fp - fx) / h
        
        if not math.isfinite(df) or abs(df) < 1e-16:
            break
        
        step = fx / df
        # Limit step
        max_step = 50.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
        
        x_new = x - step
        if not math.isfinite(x_new):
            break
        
        fx_new = safe_eval(x_new)
        if fx_new is None:
            # Try half step
            x_new = x - 0.5 * step
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break
        
        afx_new = abs(fx_new)
        update_best(x_new, fx_new)
        
        if afx_new <= 1e-8:
            return x_new
        
        # If function value increased significantly, damp
        if afx_new > abs(fx) * 2.0:
            # Bisection-like backtracking
            x_new = 0.5 * (x + x_new)
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break
            afx_new = abs(fx_new)
            update_best(x_new, fx_new)
            if afx_new <= 1e-8:
                return x_new
        
        # Check convergence
        if abs(step) < 1e-13 * max(1.0, abs(x_new)):
            newton_ok = True
            break
        
        x = x_new
        fx = fx_new
    
    if best_abs <= 1e-8:
        return best_x
    
    # Strategy 2: Find a bracket and use bisection
    a = None
    b = None
    fa = None
    fb = None
    
    # Search for sign change in multiple directions and scales
    center = best_x
    base = max(1.0, abs(center))
    scales = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    
    for scale in scales:
        if not check_time() or evals_used >= max_evals - 20:
            break
        delta = scale * base
        candidates = [(center - delta, center + delta),
                      (center - 2*delta, center + 2*delta),
                      (center, center + delta),
                      (center - delta, center)]
        for (xa, xb) in candidates:
            if not check_time() or evals_used >= max_evals - 15:
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
    
    if a is not None and b is not None and check_time() and evals_used < max_evals - 5:
        # Bisection
        fa = safe_eval(a)
        fb = safe_eval(b)
        if fa is not None and fb is not None and fa * fb < 0:
            for _ in range(80):
                if not check_time() or evals_used >= max_evals - 2:
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
    
    # Strategy 3: Secant method from best point
    if check_time() and evals_used < max_evals - 10:
        x0s = best_x
        fx0 = best_fx
        step_init = 1e-7 * max(1.0, abs(x0s))
        x1s = x0s + step_init
        fx1 = safe_eval(x1s)
        if fx1 is None:
            x1s = x0s - step_init
            fx1 = safe_eval(x1s)
        if fx1 is not None:
            update_best(x1s, fx1)
            if best_abs <= 1e-8:
                return best_x
            
            for _ in range(40):
                if not check_time() or evals_used >= max_evals - 2:
                    break
                if abs(fx1) < abs(fx0):
                    x0s, x1s = x1s, x0s
                    fx0, fx1 = fx1, fx0
                if abs(fx1) <= 1e-8:
                    return x1s
                denom = fx1 - fx0
                if abs(denom) < 1e-16:
                    break
                x2s = x1s - fx1 * (x1s - x0s) / denom
                if not math.isfinite(x2s):
                    break
                max_step = 20.0 * max(1.0, abs(x1s))
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
    
    # Strategy 4: Grid search in neighborhood if still stuck
    if check_time() and evals_used < max_evals - 20 and best_abs > 1e-10:
        center = best_x
        base = max(1.0, abs(center))
        # Try a logarithmic grid around center
        for power in range(-15, 5):
            if not check_time() or evals_used >= max_evals - 5:
                break
            delta = (10 ** power) * base
            if delta < 1e-15 * base:
                continue
            for sign in [1, -1]:
                if not check_time() or evals_used >= max_evals - 5:
                    break
                trial = center + sign * delta
                fval = safe_eval(trial)
                if fval is not None:
                    update_best(trial, fval)
                    if best_abs <= 1e-8:
                        return best_x
    
    return best_x
# EVOLVE-BLOCK-END
