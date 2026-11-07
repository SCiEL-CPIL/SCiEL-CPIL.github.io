import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 3 equations + overhead
    
    call_count = 0
    max_calls = 480  # Slightly lower to avoid edge cases where we run out of time
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
    
    # Check if x0 is already a root
    try:
        fx0 = f_counted(x0)
    except Exception:
        return x0 # Should not happen, but safe
        
    if abs(fx0) <= 1e-8:
        return x0
    
    # Strategy: Hybrid Bracketing and Interpolation
    # 1. Try to find a bracket [a, b] with f(a)*f(b) < 0
    # 2. Use Brent's method to find the root
    
    a = x0
    b = x0
    fa = fx0
    fb = fx0
    
    # Initial small steps to catch close roots
    h = 1e-3
    found_bracket = False
    
    # We will try to expand the interval [x0-h, x0+h] geometrically
    # but we already have x0. So we check x0+h and x0-h.
    
    for i in range(30):
        if time.time() > deadline:
            break
        if call_count >= max_calls - 5:
            break
            
        # Check right side
        b_candidate = x0 + h
        try:
            fb_candidate = f_counted(b_candidate)
        except Exception:
            break
            
        if abs(fb_candidate) <= 1e-8:
            return b_candidate
            
        if fa * fb_candidate < 0:
            found_bracket = True
            a, b, fa, fb = x0, b_candidate, fa, fb_candidate
            break
            
        # Check left side
        a_candidate = x0 - h
        try:
            fa_candidate = f_counted(a_candidate)
        except Exception:
            break
            
        if abs(fa_candidate) <= 1e-8:
            return a_candidate
            
        if fa_candidate * fa < 0:
            found_bracket = True
            a, b, fa, fb = a_candidate, x0, fa_candidate, fa
            break
            
        # If no bracket found yet, expand h
        # If we haven't found a bracket, we might be in a monotonic region or far from root.
        # Expand the search space.
        h *= 2.0
        
        # If h becomes too large, we might miss the root if it's not in [x0-h, x0+h]
        # but for smooth polynomials, this usually works unless the root is very far.
        
    if found_bracket:
        try:
            # Use Brent's method for high precision and low call count
            root = brentq(f_counted, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            # Verify
            if math.isfinite(root):
                return root
        except Exception:
            pass
    
    # If bracketing failed, try Secant Method
    # We have points (x0, fx0) and potentially (x0+h, fb_candidate) or (x0-h, fa_candidate)
    # Let's pick the one that gave the smaller |f|
    
    # We need two points for secant.
    # If we didn't find a bracket, we might have evaluated x0, x0+h, x0-h...
    # Let's just pick x0 and x0+1e-3 if available, or re-evaluate if needed.
    
    x1 = x0 + 1e-3
    try:
        fx1 = f_counted(x1)
    except Exception:
        # If we can't even call this, we are in trouble. Return best.
        return best_x
        
    if abs(fx1) <= 1e-8:
        return x1
    
    x_prev = x0
    f_prev = fx0
    x_curr = x1
    f_curr = fx1
    
    for _ in range(50):
        if time.time() > deadline:
            break
        if call_count >= max_calls:
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
            
    # Return the best estimate
    if math.isfinite(best_x):
        return best_x
    return x_curr
