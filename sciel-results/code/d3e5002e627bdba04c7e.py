import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    x = float(x0)
    best_x = x
    best_f = f(x)
    
    # Phase 1: Newton's Method
    # We use a limited number of steps to avoid wasting calls on divergence.
    for i in range(10):
        if time.time() > deadline:
            break
            
        if not np.isfinite(best_f):
            break
            
        if abs(best_f) <= 1e-8:
            return best_x
            
        # Finite difference derivative (forward)
        h = 1e-6 * max(1.0, abs(x))
        fx_plus = f(x + h)
        if time.time() > deadline:
            break
        if not np.isfinite(fx_plus):
            break
            
        d = (fx_plus - best_f) / h
        
        if d == 0 or not np.isfinite(d):
            break
            
        x_new = x - best_f / d
        
        # Check for divergence or oscillation
        if not np.isfinite(x_new):
            break
            
        # If the step is too large, we might be far away, but let's try once
        # If it oscillates, we will catch it in the bracketing phase
        
        x = x_new
        best_f = f(x)
        
        if abs(best_f) <= 1e-8:
            return x
            
        # Heuristic: if we are moving away from the initial guess significantly 
        # and not converging, we might be stuck. But let's let the loop finish 
        # or rely on the bracketing fallback.
        
    # Phase 2: Bracketing and Brent's Method
    # If Newton didn't find the root, we need a bracket.
    # We use the current x (or x0 if x is bad) as the center.
    center = x if np.isfinite(x) and abs(f(x)) < abs(f(x0)) else x0
    # Actually, let's just use x0 as the center for bracketing if Newton failed,
    # or the last x. The last x might be closer. Let's try center = x.
    # But if x diverged to infinity, f(x) might be garbage.
    # Let's use the point with smallest |f| seen so far.
    
    # Re-evaluate f at x0 to have a baseline if needed? No, we have best_f from loop.
    # But if loop broke immediately, best_f is f(x0).
    
    # Let's try to find a bracket around center
    a = center
    b = center
    fa = f(a)
    fb = f(b)
    
    # We need distinct a and b.
    # Start with a small step
    step = 1.0
    
    # If center is 0, step 1 is fine. If center is large, step should be relative.
    step = max(1.0, 0.1 * abs(center))
    
    found_bracket = False
    # Expand outward
    for _ in range(50):
        if time.time() > deadline:
            break
            
        a = center - step
        b = center + step
        
        fa = f(a)
        fb = f(b)
        
        if not np.isfinite(fa) or not np.isfinite(fb):
            step *= 2
            continue
            
        if fa * fb < 0:
            found_bracket = True
            break
            
        step *= 2
        
    if found_bracket:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if np.isfinite(root):
                return root
        except Exception:
            pass
            
    # Fallback: Return the best point found
    return best_x

# EVOLVE-BLOCK-END
