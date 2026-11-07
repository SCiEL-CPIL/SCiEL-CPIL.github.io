import math
import time
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # Safety margin
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # Phase 1: Find a bracket
    # Try expanding from x0
    # We want to find a, b such that f(a)*f(b) < 0
    # Start with a small step
    step = 1e-3
    max_step = 1000.0
    
    a = x
    fa = fx
    b = None
    fb = None
    
    # Try to find a sign change
    # Check left and right
    # To save calls, we can try a few steps in both directions
    # But usually, one direction works.
    
    # Let's try a few small steps first
    candidates = []
    for i in range(1, 10):
        candidates.append(x + step * i)
        candidates.append(x - step * i)
        
    # Actually, expanding sequentially is safer to maintain continuity for brentq
    # Let's expand right first, then left if needed? Or alternate?
    # Alternating is better to stay close to x0.
    
    left = x
    right = x
    f_left = fx
    f_right = fx
    
    # We already have f(x).
    # Try x + step, x - step, x + 2*step, x - 2*step...
    
    found_bracket = False
    bracket_a = None
    bracket_b = None
    
    # We need to be careful with call count.
    # Each new point is 1 call.
    # If we find a bracket, we call brentq.
    # brentq will make more calls.
    # Total calls = (expansion calls) + (brentq calls).
    # Expansion: ~5-10 calls. Brentq: ~10-20 calls. Total < 30. Well under 500.
    
    for i in range(1, 20):
        # Check time
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Try right
        x_r = x + step * i
        f_r = f(x_r)
        if abs(f_r) <= 1e-8:
            return x_r
        
        if f_r * fx < 0:
            bracket_a = x
            bracket_b = x_r
            found_bracket = True
            break
            
        # Try left
        x_l = x - step * i
        f_l = f(x_l)
        if abs(f_l) <= 1e-8:
            return x_l
            
        if f_l * fx < 0:
            bracket_a = x_l
            bracket_b = x
            found_bracket = True
            break
            
        # Update boundaries for potential future use if we switch to minimization
        if abs(f_r) < abs(f_left):
            f_left = f_r
            left = x_r
        if abs(f_l) < abs(f_right):
            f_right = f_l
            right = x_l
            
        # Increase step size? No, linear expansion is fine for oscillatory.
        # If step is too small, we might take many calls.
        # If step is too large, we might skip over a root.
        # For oscillatory, roots are spaced.
        # Let's stick to linear expansion with a reasonable base step.
        # If the function is very oscillatory, step=1e-3 might be too small.
        # If it's very flat, step=1e-3 might be too big?
        # Let's try adaptive step?
        # Simpler: just try a range of steps.
        
        # Actually, let's just try a few specific offsets if the simple loop doesn't work.
        # But the loop above does that.
        
        # If we haven't found a bracket, maybe the root is further.
        # We can increase the base step.
        if i == 5:
            step *= 10
        elif i == 10:
            step *= 10

    if found_bracket:
        try:
            # Use brentq
            # maxiter=100 is plenty
            root = brentq(f, bracket_a, bracket_b, maxiter=100, xtol=1e-12, rtol=1e-12)
            # Verify
            if abs(f(root)) <= 1e-8:
                return root
        except Exception:
            pass

    # Phase 2: Fallback to local minimization
    # Minimize f(x)^2 in a neighborhood
    # Choose a window around x0
    # If we didn't find a bracket, the function might not change sign.
    # Or the root is far.
    # Let's try minimizing in [x0 - 1, x0 + 1], [x0 - 10, x0 + 10], etc.
    
    # Re-use some calls? We already have f at several points.
    # minimize_scalar doesn't reuse easily unless we wrap f.
    # But we can just call it.
    
    def obj(x):
        return f(x)**2
        
    # Try a few windows
    for width in [1e-2, 1e-1, 1, 10, 100]:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        a_min = x - width
        b_min = x + width
        
        try:
            res = minimize_scalar(obj, bounds=(a_min, b_min), method='bounded', options={'xatol': 1e-12})
            if res.success and res.fun <= 1e-16: # 1e-8 squared
                return res.x
        except Exception:
            pass

    # If all else fails, return the best point found during expansion
    # We tracked left/right with min |f|?
    # Actually, we didn't track the global min |f| rigorously in the loop above.
    # Let's just return x0 or the closest point we evaluated.
    
    # Better: Do a final check on the points we evaluated.
    # We didn't store them all.
    # Let's just return x0. It's a failure if |f(x0)| > 1e-8, but it's better than an exception.
    
    return x
