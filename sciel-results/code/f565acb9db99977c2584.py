import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 3 equations + overhead
    
    call_count = 0
    max_calls = 490  # Safe margin below 500
    best_x = x0
    best_f = float('inf')
    
    def f_counted(x):
        nonlocal call_count, best_x, best_f
        if call_count >= max_calls:
            raise Exception("Max calls exceeded")
        call_count += 1
        val = f(x)
        if math.isfinite(val) and abs(val) < abs(best_f):
            best_f = val
            best_x = x
        return val

    # 1. Check if x0 is already a root
    try:
        fx0 = f_counted(x0)
    except Exception:
        return x0
        
    if abs(fx0) <= 1e-8:
        return x0

    # 2. Strategy: Cubic Reconstruction (Single-Shot)
    # The task specifies "3 fixed smooth polynomial equations".
    # If the degree is <= 3, 4 points determine the polynomial exactly.
    # Even for higher degrees, if the points are close enough or the function
    # is well-behaved, the root of the interpolating cubic might be very close.
    # This requires only 4 calls (plus 1 for verification).
    
    # Choose points. We want them distinct and reasonably spaced.
    # If x0 is large, we need to be careful with floating point precision.
    # Let's try a symmetric set around x0 first, or just offsets.
    # Offsets: 0, 1, 2, 3 relative to x0? 
    # Or maybe 0, 0.5, 1.0, 1.5?
    # Let's try [x0, x0+1, x0+2, x0+3] first.
    # If x0 is huge, x0+1 might equal x0 in float64.
    # Check for distinctness.
    
    points = [x0, x0 + 1.0, x0 + 2.0, x0 + 3.0]
    
    # Ensure points are distinct
    distinct = True
    for i in range(4):
        for j in range(i+1, 4):
            if points[i] == points[j]:
                distinct = False
                break
        if not distinct:
            break
            
    if not distinct:
        # If x0 is very large, 1.0 is too small. Try smaller offsets?
        # Or larger? If x0 is 1e16, 1.0 is lost. 
        # Actually, if x0 is that large, the root is likely not near x0 unless f is weird.
        # Let's try scaling based on magnitude.
        scale = max(1.0, abs(x0) * 1e-3)
        points = [x0, x0 + scale, x0 + 2*scale, x0 + 3*scale]
        
        distinct = True
        for i in range(4):
            for j in range(i+1, 4):
                if points[i] == points[j]:
                    distinct = False
                    break
            if not distinct:
                break
        
        if not distinct:
            # Fallback: just use x0 and x0+epsilon
            # But we need 4 points. This is a rare edge case.
            # Let's just proceed to bracketing if this fails.
            pass
    else:
        try:
            if time.time() > deadline:
                return best_x
            
            vals = []
            for p in points:
                if time.time() > deadline or call_count >= max_calls - 5:
                    raise Exception("Time or calls running out")
                v = f_counted(p)
                vals.append(v)
                if abs(v) <= 1e-8:
                    return p
            
            # Interpolate cubic polynomial
            # np.polyfit returns coefficients [c3, c2, c1, c0] for c3*x^3 + ...
            coeffs = np.polyfit(points, vals, 3)
            roots = np.roots(coeffs)
            
            # Filter for real roots
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
            
            if real_roots:
                # Find the root closest to x0
                best_root = min(real_roots, key=lambda r: abs(r - x0))
                
                # Verify the root
                if time.time() > deadline or call_count >= max_calls:
                    return best_root
                    
                f_best = f_counted(best_root)
                if abs(f_best) <= 1e-8:
                    return best_root
                else:
                    # Cubic approximation failed (likely degree > 3 or ill-conditioned)
                    # We have used 5 calls (4 points + 1 verify).
                    # Proceed to robust methods.
                    pass
        except Exception:
            pass

    # 3. Strategy: Bracketing with Brent's Method
    # If cubic failed, we need a robust method.
    # We have evaluated x0, and possibly x0+1, x0+2, x0+3.
    # Let's reuse those evaluations if possible? 
    # f_counted doesn't cache, so we'd have to re-evaluate.
    # However, we can check the sign of the points we already know about.
    # But we don't have the values stored in a way that's easy to access without re-calling.
    # Actually, we can just start a fresh bracketing search.
    # Since we have ~480 calls left, we can afford a geometric expansion.
    
    a = x0
    fa = fx0
    
    # Try expanding to the right first, then left.
    # Start with a small step to catch close roots, then expand.
    step = 1e-3
    found_bracket = False
    last_a, last_b, last_fa, last_fb = None, None, None, None
    
    for i in range(50):
        if time.time() > deadline or call_count > max_calls - 10:
            break
            
        # Try right
        b = x0 + step
        try:
            fb = f_counted(b)
        except Exception:
            break
            
        if abs(fb) <= 1e-8:
            return b
            
        if fa * fb < 0:
            found_bracket = True
            last_a, last_b, last_fa, last_fb = x0, b, fa, fb
            break
            
        # Try left
        a_left = x0 - step
        try:
            fa_left = f_counted(a_left)
        except Exception:
            break
            
        if abs(fa_left) <= 1e-8:
            return a_left
            
        if fa_left * fa < 0:
            found_bracket = True
            last_a, last_b, last_fa, last_fb = a_left, x0, fa_left, fa
            break
            
        step *= 2.0
        
        # If step becomes too large, we might jump over the root.
        # But for polynomials, this is less likely unless the root is extremely far.
        
    if found_bracket:
        try:
            root = brentq(f_counted, last_a, last_b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass

    # 4. Strategy: Secant Method
    # If bracketing failed, try secant.
    # We need two points. We have x0. Let's get x0 + 1e-4.
    
    x1 = x0 + 1e-4
    try:
        fx1 = f_counted(x1)
    except Exception:
        return best_x
        
    if abs(fx1) <= 1e-8:
        return x1
        
    x_prev = x0
    f_prev = fx0
    x_curr = x1
    f_curr = fx1
    
    for _ in range(50):
        if time.time() > deadline or call_count > max_calls - 1:
            break
            
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            # Parallel secants, try a small perturbation
            x_new = x_curr + 1e-6
        else:
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            
        if not math.isfinite(x_new):
            break
            
        # Damping if step is huge
        if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
            x_new = x_curr + 10 * max(1.0, abs(x_curr)) * (1 if x_new > x_curr else -1)
            
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        
        try:
            f_curr = f_counted(x_curr)
        except Exception:
            break
            
        if abs(f_curr) <= 1e-8:
            return x_curr
            
    # Return the best estimate
    if math.isfinite(best_x):
        return best_x
    return x_curr
