"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin before 150s kill
    x0 = float(x0)
    
    # Check initial point
    f0 = f(x0)
    if not np.isfinite(f0):
        f0 = float('inf')
    if abs(f0) <= 1e-8:
        return x0
    
    best_x = x0
    best_f = abs(f0)
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT
    
    # Geometric expansion to find a bracket
    # Start with a step size relative to x0
    h = max(1.0, 0.1 * abs(x0))
    
    # We will check intervals [x0 - h*2^k, x0 + h*2^k]
    # But more efficiently, we can just check points x0 +/- h*2^k
    # and look for sign changes between consecutive points.
    
    prev_x_pos = x0
    prev_f_pos = f0
    prev_x_neg = x0
    prev_f_neg = f0
    
    bracket_found = False
    a = 0.0
    b = 0.0
    
    # Expand up to 30 steps (h * 2^30 is huge, usually enough)
    # Limit calls: 1 (initial) + 2 * 30 = 61 calls for expansion.
    for k in range(30):
        if not time_ok():
            break
        
        # Positive direction
        x_pos = x0 + h * (2 ** k)
        f_pos = f(x_pos)
        
        if np.isfinite(f_pos):
            if abs(f_pos) < best_f:
                best_f = abs(f_pos)
                best_x = x_pos
            
            if abs(f_pos) <= 1e-8:
                return x_pos
            
            # Check sign change between prev_x_pos and x_pos
            if prev_f_pos * f_pos < 0:
                a, b = prev_x_pos, x_pos
                bracket_found = True
                break
            
            prev_x_pos = x_pos
            prev_f_pos = f_pos
        
        # Negative direction
        x_neg = x0 - h * (2 ** k)
        f_neg = f(x_neg)
        
        if np.isfinite(f_neg):
            if abs(f_neg) < best_f:
                best_f = abs(f_neg)
                best_x = x_neg
            
            if abs(f_neg) <= 1e-8:
                return x_neg
            
            # Check sign change between prev_x_neg and x_neg
            if prev_f_neg * f_neg < 0:
                a, b = x_neg, prev_x_neg
                bracket_found = True
                break
            
            prev_x_neg = x_neg
            prev_f_neg = f_neg
        
        # If we've expanded a lot, maybe stop to save calls for refinement
        if k > 20:
            break
    
    if bracket_found:
        try:
            # Use brentq for high precision and efficiency
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if np.isfinite(root):
                return root
        except Exception:
            pass
    
    # Fallback: If no bracket found, return best_x found
    # We could try a local minimization or Newton here, but risk is high.
    # Given the score, just returning best_x is safer than diverging Newton.
    return best_x
# EVOLVE-BLOCK-END
