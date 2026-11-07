import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 118.0
    
    call_count = 0
    max_calls = 500
    best_x = x0
    best_f = float('inf')
    
    def f_counted(x):
        nonlocal call_count, best_x, best_f
        if call_count >= max_calls:
            raise Exception("Max calls exceeded")
        call_count += 1
        val = f(x)
        if abs(val) < abs(best_f):
            best_f = val
            best_x = x
        return val
    
    def check_root(x, fx):
        nonlocal best_x, best_f
        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        if abs(fx) <= 1e-8:
            return True
        return False
    
    # Check if x0 is already a root
    try:
        fx0 = f_counted(x0)
    except Exception:
        return x0
    if check_root(x0, fx0):
        return x0
    
    # Strategy 1: Cubic reconstruction with adaptive points
    # Use points that are more likely to bracket the root
    candidates = []
    
    try:
        # Try several sets of 4 points, starting with tight spacing
        point_sets = [
            [x0, x0 + 0.5, x0 + 1.0, x0 + 2.0],
            [x0 - 1.0, x0 - 0.5, x0 + 0.5, x0 + 1.0],
            [x0, x0 + 0.1, x0 + 0.5, x0 + 1.0],
            [x0 - 0.5, x0, x0 + 0.5, x0 + 1.0],
        ]
        
        for pts in point_sets:
            if call_count >= max_calls - 10 or time.time() > deadline:
                break
            
            vals = []
            all_distinct = True
            for i, p in enumerate(pts):
                if not math.isfinite(p):
                    all_distinct = False
                    break
                if i > 0 and p == pts[i-1]:
                    all_distinct = False
                    break
                v = f_counted(p)
                if not math.isfinite(v):
                    all_distinct = False
                    break
                vals.append(v)
                if check_root(p, v):
                    return p
            
            if not all_distinct or len(vals) < 4:
                continue
            
            # Fit cubic
            try:
                coeffs = np.polyfit(pts, vals, 3)
                roots = np.roots(coeffs)
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and math.isfinite(r.real)]
            except Exception:
                continue
            
            if not real_roots:
                continue
            
            # Sort roots by distance to x0
            real_roots.sort(key=lambda r: abs(r - x0))
            
            # Try each real root as candidate, verify and polish
            for root in real_roots[:2]:
                if call_count >= max_calls - 5 or time.time() > deadline:
                    break
                
                try:
                    fr = f_counted(root)
                    if check_root(root, fr):
                        return root
                    
                    # If close, try to polish with secant
                    if abs(fr) < 1e-4:
                        # Use secant with two nearby points
                        h = 1e-6
                        x1 = root - h
                        x2 = root + h
                        if call_count < max_calls - 2:
                            f1 = f_counted(x1)
                            if check_root(x1, f1):
                                return x1
                            f2 = f_counted(x2)
                            if check_root(x2, f2):
                                return x2
                            
                            # Secant step
                            denom = f2 - f1
                            if abs(denom) > 1e-15:
                                x_new = x2 - f2 * (x2 - x1) / denom
                                if math.isfinite(x_new) and call_count < max_calls:
                                    f_new = f_counted(x_new)
                                    if check_root(x_new, f_new):
                                        return x_new
                except Exception:
                    continue
            
            # If we found any real roots, we've used our cubic budget
            break
    
    except Exception:
        pass
    
    # Strategy 2: Bracketing with Brent's method
    # Use multiple starting directions
    fa = fx0
    found_bracket = False
    bracket_a = x0
    bracket_b = None
    bracket_fb = None
    
    # Try expanding in both directions with adaptive steps
    step = 0.1
    for direction in [1, -1]:
        if call_count >= max_calls - 10 or time.time() > deadline:
            break
        
        for i in range(30):
            if call_count >= max_calls - 10 or time.time() > deadline:
                break
            
            b = x0 + direction * step
            if not math.isfinite(b):
                break
            
            try:
                fb = f_counted(b)
            except Exception:
                break
            
            if check_root(b, fb):
                return b
            
            if fa * fb < 0:
                found_bracket = True
                bracket_a = x0
                bracket_b = b
                bracket_fb = fb
                break
            
            step *= 2.0
    
    if found_bracket:
        try:
            root = brentq(f_counted, bracket_a, bracket_b, xtol=1e-13, rtol=1e-13, maxiter=50)
            if math.isfinite(root):
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    return root
                # Polish
                if abs(fr) < 1e-6 and call_count < max_calls - 2:
                    # Secant polish
                    h = 1e-8
                    x1 = root - h
                    x2 = root + h
                    f1 = f_counted(x1)
                    f2 = f_counted(x2)
                    if abs(f2 - f1) > 1e-15:
                        x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                        if math.isfinite(x_new) and call_count < max_calls:
                            f_new = f_counted(x_new)
                            if abs(f_new) <= 1e-8:
                                return x_new
        except Exception:
            pass
    
    # Strategy 3: Secant method with multiple starting points
    # Use the best points we've seen so far
    # Try secant from x0 with a small perturbation
    x1 = x0 + 1e-4
    if call_count < max_calls - 1:
        try:
            fx1 = f_counted(x1)
        except Exception:
            fx1 = fx0
    else:
        fx1 = fx0
    
    x_prev = x0
    x_curr = x1
    f_prev = fx0
    f_curr = fx1
    
    for _ in range(40):
        if time.time() > deadline or call_count > max_calls - 1:
            break
        
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            break
            
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_new):
            break
            
        # Clamp to reasonable range
        if abs(x_new - x_curr) > 100 * max(1.0, abs(x_curr)):
            x_new = x_curr + 100 * max(1.0, abs(x_curr)) * (1 if x_new > x_curr else -1)
            
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        
        if call_count < max_calls:
            try:
                f_curr = f_counted(x_curr)
            except Exception:
                break
        else:
            break
            
        if abs(f_curr) <= 1e-8:
            return x_curr
    
    # Return the best point found
    return best_x
