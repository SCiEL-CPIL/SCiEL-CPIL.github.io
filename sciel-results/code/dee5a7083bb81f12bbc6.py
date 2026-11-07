import math
import time
import sys
from scipy.optimize import brentq, minimize

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 440
    DEADLINE = time.time() + 130
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = x
            if abs_val <= TOL:
                return val
            return val
        except Exception:
            return float('inf')

    def obj_abs(x):
        """Objective for Nelder-Mead: minimize |f(x)|"""
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            return float('inf')
        val = safe_f(x[0])
        if math.isinf(val):
            return float('inf')
        return abs(val)

    # Initial evaluation
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    
    # Bracketing search with adaptive multi-scale approach
    # First try fine scales, then expand
    base_scale = max(1e-3, abs(x0) * 1e-3)
    scales = []
    for i in range(12):
        s = base_scale * (10 ** i)
        if s > 1000:
            break
        scales.append(s)
    
    bracket_found = False
    a = None
    b = None
    
    # Try to find bracket by checking sign changes at multiple scales
    for scale in scales:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        for mult in [0.5, 1.0, 2.0, 5.0]:
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            d = mult * scale
            for sign in [1, -1]:
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                x_new = x0 + sign * d
                f_new = safe_f(x_new)
                
                if math.isinf(f_new):
                    continue
                
                # Check if we have a bracket with x0
                if fx0 * f_new < 0:
                    a = x0
                    b = x_new
                    bracket_found = True
                    break
                
                # Check if we have a bracket with a previous point
                if a is not None and b is None:
                    if safe_f(a) * f_new < 0:
                        b = x_new
                        bracket_found = True
                        break
                    elif b is None and math.isfinite(fx0):
                        a = x_new
                        b = None
            
            if bracket_found:
                break
        
        if bracket_found:
            break
    
    # If we have a bracket, use brentq
    if bracket_found and a is not None and b is not None:
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=60)
            if math.isfinite(root) and abs(safe_f(root)) <= TOL:
                return root
        except Exception:
            pass
    
    # If no bracket found, try Nelder-Mead on |f(x)|
    if not bracket_found and calls < MAX_CALLS - 50 and time.time() < DEADLINE - 20:
        try:
            # Initialize simplex around best_x
            x_init = [best_x]
            h = max(1e-3, abs(best_x) * 1e-3)
            x_init2 = [best_x + h]
            
            res = minimize(obj_abs, x_init, method='Nelder-Mead', 
                          options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 200, 'maxfev': 100})
            if math.isfinite(res.x[0]):
                candidate = res.x[0]
                fcandidate = safe_f(candidate)
                if math.isfinite(fcandidate) and abs(fcandidate) <= TOL:
                    return candidate
                elif math.isfinite(fcandidate) and abs(fcandidate) < best_abs:
                    best_x = candidate
                    best_abs = abs(fcandidate)
        except Exception:
            pass
    
    # Newton's method fallback from the best point found
    x = best_x
    for _ in range(40):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        
        h = 1e-6 * max(1.0, abs(x))
        f_plus = safe_f(x + h)
        if math.isinf(f_plus):
            # Try other direction
            f_minus = safe_f(x - h)
            if math.isinf(f_minus):
                break
            d = (fx - f_minus) / h
        else:
            d = (f_plus - fx) / h
        
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
            
        # Damping if step is too large
        max_step = 5 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
             x_new = x + max_step * (1 if x_new > x else -1)
             
        x = x_new
        
    return best_x
