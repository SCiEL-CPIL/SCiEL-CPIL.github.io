import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    """
    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0

    best_x = x0
    best_f = None
    try:
        best_f = f(x0)
    except:
        best_f = float('inf')

    if best_f is not None and math.isfinite(best_f) and abs(best_f) <= 1e-8:
        return x0

    # Strategy 1: Find a bracket and use brentq
    # Expand steps exponentially in both directions
    steps = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    
    # Store current best
    if best_f is None or not math.isfinite(best_f):
         best_f = float('inf')

    for direction in [1, -1]:
        if _elapsed() > 140.0: # Leave 10s margin
            break
            
        prev_x = x0
        prev_f = best_f
        if prev_f is None or not math.isfinite(prev_f):
             # If x0 is bad, try a small step first to get a valid point
             # We will handle this in the loop by checking validity
             pass

        for s in steps:
            if _elapsed() > 140.0:
                break
                
            target_x = x0 + direction * s
            
            try:
                target_f = f(target_x)
            except:
                target_f = float('inf')
            
            if not math.isfinite(target_f):
                continue

            # Update best
            if abs(target_f) < abs(best_f):
                best_f = target_f
                best_x = target_x
                
            if abs(target_f) <= 1e-8:
                return target_x

            # Check for sign change between prev_x and target_x
            # We need prev_f to be finite. If prev_x was x0 and x0 was invalid, we can't bracket.
            # So we should ensure we have a valid prev_f.
            
            if math.isfinite(prev_f):
                if prev_f * target_f < 0:
                    # Bracket found
                    try:
                        root = brentq(f, prev_x, target_x, xtol=1e-12, rtol=1e-12, maxiter=100)
                        if math.isfinite(root):
                            return root
                    except:
                        pass
                elif prev_f == 0:
                    return prev_x
                elif target_f == 0:
                    return target_x

            # Update prev for next step
            prev_x = target_x
            prev_f = target_f

    # Strategy 2: Local minimization of f(x)^2
    # If no bracket found, the root might be tangent or the function doesn't change sign.
    # Use minimize_scalar on a bounded interval around x0.
    
    # Determine a reasonable bound
    # The steps we tried went up to 500. Let's search in [-1000, 1000] relative to x0?
    # Or just use the range we explored.
    # Let's try a few centered intervals.
    
    bounds_list = [
        (x0 - 1.0, x0 + 1.0),
        (x0 - 10.0, x0 + 10.0),
        (x0 - 100.0, x0 + 100.0),
        (x0 - 1000.0, x0 + 1000.0)
    ]

    for a, b in bounds_list:
        if _elapsed() > 145.0:
            break
            
        # Ensure a < b
        if a > b:
            a, b = b, a
            
        try:
            # minimize_scalar on f(x)^2
            # bounds method requires a finite function
            def squared_residual(x):
                try:
                    val = f(x)
                    if not math.isfinite(val):
                        return 1e100
                    return val * val
                except:
                    return 1e100

            res = minimize_scalar(squared_residual, bounds=(a, b), method='bounded', options={'maxiter': 100, 'xatol': 1e-12})
            if res.success or math.isfinite(res.fun):
                x_candidate = res.x
                try:
                    f_candidate = f(x_candidate)
                except:
                    f_candidate = float('inf')
                
                if math.isfinite(f_candidate):
                    if abs(f_candidate) < abs(best_f):
                        best_f = f_candidate
                        best_x = x_candidate
                        
                    if abs(f_candidate) <= 1e-8:
                        return x_candidate
        except:
            continue

    return best_x
