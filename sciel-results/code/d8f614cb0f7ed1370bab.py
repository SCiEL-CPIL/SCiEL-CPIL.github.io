import math
import time
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses a robust multi-scale bracketing search followed by Brent's method.
    Falls back to scalar minimization of f(x)^2 if no bracket is found.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit
    max_evals = 500
    eval_count = 0
    
    def check_time():
        return time.time() > deadline

    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        try:
            fx = f(x)
            eval_count += 1
            if math.isnan(fx) or math.isinf(fx):
                return None
            return fx
        except Exception:
            return None

    # Initial evaluation
    x0 = float(x0)
    fx0 = eval_f(x0)
    if fx0 is None:
        return x0
        
    best_x = x0
    best_fx = fx0
    
    if abs(fx0) <= 1e-8:
        return x0

    scale = max(1.0, abs(x0))
    
    # Strategy 1: Robust Bracketing Search
    # Try a range of step sizes to find a sign change
    # Geometric progression of steps
    steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    a, b = None, None
    fa, fb = None, None
    bracket_found = False
    
    for s in steps:
        if check_time() or eval_count >= max_evals - 10:
            break
            
        delta = s * scale
        
        # Try right
        x_r = x0 + delta
        fx_r = eval_f(x_r)
        if fx_r is not None:
            if abs(fx_r) < abs(best_fx):
                best_x, best_fx = x_r, fx_r
            if abs(fx_r) <= 1e-8:
                return x_r
            if fx0 * fx_r < 0:
                a, b = x0, x_r
                fa, fb = fx0, fx_r
                bracket_found = True
                break
        
        if check_time() or eval_count >= max_evals - 10:
            break
            
        # Try left
        x_l = x0 - delta
        fx_l = eval_f(x_l)
        if fx_l is not None:
            if abs(fx_l) < abs(best_fx):
                best_x, best_fx = x_l, fx_l
            if abs(fx_l) <= 1e-8:
                return x_l
            if fx0 * fx_l < 0:
                a, b = x_l, x0
                fa, fb = fx_l, fx0
                bracket_found = True
                break
        
        # Check if fx_r and fx_l bracket (if both evaluated)
        if fx_r is not None and fx_l is not None and fx_r * fx_l < 0:
            a, b = x_l, x_r
            fa, fb = fx_l, fx_r
            bracket_found = True
            break

    if bracket_found:
        # Use Brent's method, which is robust and fast
        try:
            # brentq requires f(a) and f(b) to have different signs
            # We already checked this, but ensure no zero division issues or NaNs
            if fa * fb < 0:
                # xtol and rtol are relative to the bracket size
                # We want |f(x)| <= 1e-8, so we need high precision
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100, full_output=False)
                # Verify
                fr = eval_f(root)
                if fr is not None and abs(fr) <= 1e-8:
                    return root
                elif fr is not None and abs(fr) < abs(best_fx):
                    best_x, best_fx = root, fr
        except Exception:
            pass

    # Strategy 2: Minimize f(x)^2 to find a local minimum
    # This is useful if the root is a "touching" root or if bracketing failed
    if eval_count < max_evals - 10 and not check_time():
        def obj(x):
            fx = eval_f(x)
            if fx is None:
                return 1e100
            return fx * fx
        
        # Use bounded minimization around x0
        # The bound is somewhat arbitrary, but we can use the largest step tried
        bound = 10000.0 * scale
        
        try:
            res = minimize_scalar(obj, bounds=(x0 - bound, x0 + bound), method='bounded', options={'xatol': 1e-12})
            if res.success:
                x_min = res.x
                f_min = eval_f(x_min)
                if f_min is not None:
                    if abs(f_min) <= 1e-8:
                        return x_min
                    elif abs(f_min) < abs(best_fx):
                        best_x, best_fx = x_min, f_min
        except Exception:
            pass

    # Strategy 3: Secant method from best_x
    # Try a few secant steps if we haven't found a root
    if eval_count < max_evals - 5 and not check_time():
        # Use best_x and a nearby point
        x_prev = best_x
        fx_prev = best_fx
        # Perturb slightly
        x_curr = best_x + 1e-4 * scale
        fx_curr = eval_f(x_curr)
        
        if fx_curr is not None and fx_curr != fx_prev:
            for _ in range(10):
                if check_time() or eval_count >= max_evals - 2:
                    break
                if abs(fx_curr - fx_prev) < 1e-15:
                    break
                x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(best_fx):
                    best_x, best_fx = x_new, fx_new
                if abs(fx_new) <= 1e-8:
                    return x_new
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = x_new, fx_new

    return best_x
