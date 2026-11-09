import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    x = float(x0)
    try:
        fx = f(x)
    except Exception:
        return x

    if not math.isfinite(fx):
        return x
        
    if abs(fx) <= 1e-8:
        return x

    # Strategy:
    # 1. Try to bracket with small steps.
    # 2. If bracketing fails, try local minimization of f(x)^2.
    # 3. Expand and repeat.
    
    # We will keep track of the best candidate found so far
    best_x = x
    best_f = abs(fx)
    
    # Initial window size for minimization and bracketing
    # Scale with x magnitude
    base_width = 0.1 * max(1.0, abs(x))
    
    # Limit total calls to stay well under 500
    # minimize_scalar bounded usually takes ~10-15 calls
    # brentq takes ~10-20 calls
    # We can afford ~20-30 attempts if they fail fast, but we should be efficient.
    
    width = base_width
    max_width = 1e6 * max(1.0, abs(x))
    
    # To avoid infinite loops or too many calls, we limit the number of expansions
    # Each expansion doubles the width.
    # Log2(1e6) is ~20. So 20 expansions max.
    # Each expansion might cost ~15 calls (minimization) or a few calls (bracketing check).
    # 20 * 15 = 300 calls. This is safe.
    
    for i in range(25): # Max 25 expansions
        # 1. Attempt to find a sign change in [x-width, x+width]
        # We sample a few points to check for sign changes cheaply
        # If we find a sign change, we use brentq on that sub-interval.
        
        # Let's sample 5 points in the window to check for sign changes
        # This costs 5 evaluations.
        # If we find a sign change, we refine with brentq.
        
        # Points: x - width, x - 0.5*width, x, x + 0.5*width, x + width
        # We already have f(x).
        # Let's check neighbors.
        
        # Instead of sampling, let's just try to bracket by expanding from x.
        # The current program does this. Let's stick to a similar bracketing logic 
        # but add the minimization fallback.
        
        # Check immediate neighbors for sign change
        x_minus = x - width
        x_plus = x + width
        
        try:
            fx_minus = f(x_minus)
            fx_plus = f(x_plus)
        except Exception:
            fx_minus = float('nan')
            fx_plus = float('nan')
            
        found_bracket = False
        a = None
        b = None
        
        if math.isfinite(fx_minus) and math.isfinite(fx):
            if fx * fx_minus < 0:
                a, b = x_minus, x
                found_bracket = True
            if abs(fx_minus) < best_f:
                best_x = x_minus
                best_f = abs(fx_minus)
                
        if math.isfinite(fx_plus) and math.isfinite(fx):
            if fx * fx_plus < 0:
                a, b = x, x_plus
                found_bracket = True
            if abs(fx_plus) < best_f:
                best_x = x_plus
                best_f = abs(fx_plus)
                
        if found_bracket:
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                if math.isfinite(root):
                    fr = f(root)
                    if math.isfinite(fr) and abs(fr) <= 1e-8:
                        return root
                    # If brentq succeeded but tolerance wasn't met (unlikely with 1e-12), 
                    # return root anyway as it's likely close.
                    return root
            except Exception:
                pass

        # 2. If no bracket found, try minimization of f(x)^2 in [x-width, x+width]
        # This handles tangent roots or cases where sign change is missed due to oscillation period.
        
        def objective(x_val):
            try:
                val = f(x_val)
                if not math.isfinite(val):
                    return 1e100
                return val * val
            except Exception:
                return 1e100

        try:
            res = minimize_scalar(objective, bounds=(x - width, x + width), method='bounded', options={'xatol': 1e-12})
            if res.success:
                x_min = res.x
                f_min = f(x_min)
                if math.isfinite(f_min):
                    if abs(f_min) < best_f:
                        best_x = x_min
                        best_f = abs(f_min)
                    if abs(f_min) <= 1e-8:
                        return x_min
        except Exception:
            pass

        # Expand window
        width *= 2.0
        if width > max_width:
            break
            
    # If we didn't find a root within the tolerance, return the best candidate found
    # which minimizes |f(x)|.
    return best_x
