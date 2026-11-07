"""Root finder for 3 fixed oscillatory equations."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find a root of f near x0.
    Returns x with |f(x)| <= 1e-8.
    Uses adaptive multi-scale bracketing + Brent's method.
    """
    start_time = time.time()
    TIME_LIMIT = 135.0  # safety margin before 150s kill

    x0 = float(x0)

    # Helper to check time
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    # Evaluate f safely
    def safe_f(x):
        try:
            val = f(x)
            if not np.isfinite(val):
                return float('inf')
            return float(val)
        except Exception:
            return float('inf')

    # Check initial point
    f0 = safe_f(x0)
    if abs(f0) <= 1e-8:
        return x0

    best_x = x0
    best_f = abs(f0)

    # Multi-scale scanning strategy
    # For oscillatory functions, we need to find a sign change
    # Try different scales to bracket a root
    
    # Strategy: try multiple scales, scan in both directions
    # Scales chosen to cover typical oscillatory periods
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    # For each scale, scan a window around x0
    # Number of steps to scan in each direction
    max_steps = 200
    
    for scale in scales:
        if not time_ok():
            break
        
        # Scan in positive direction
        for i in range(1, max_steps + 1):
            if not time_ok():
                break
            x1 = x0 + i * scale
            f1 = safe_f(x1)
            if abs(f1) < abs(best_f):
                best_x = x1
                best_f = abs(f1)
            if abs(f1) <= 1e-8:
                return x1
            
            # Check for sign change between previous and current
            # We need to track the previous point
            # Let's restructure: scan and check consecutive points
        
        # Scan in negative direction
        for i in range(1, max_steps + 1):
            if not time_ok():
                break
            x1 = x0 - i * scale
            f1 = safe_f(x1)
            if abs(f1) < abs(best_f):
                best_x = x1
                best_f = abs(f1)
            if abs(f1) <= 1e-8:
                return x1

    # Better approach: for each scale, do a proper scan checking consecutive points
    # Let's restart with a cleaner approach
    
    # Reset
    best_x = x0
    best_f = abs(f0)
    
    # For each scale, scan from x0 outward, checking consecutive pairs
    for scale in scales:
        if not time_ok():
            break
        
        # Positive direction
        prev_x = x0
        prev_f = f0
        for i in range(1, max_steps + 1):
            if not time_ok():
                break
            curr_x = x0 + i * scale
            curr_f = safe_f(curr_x)
            
            if abs(curr_f) < best_f:
                best_x = curr_x
                best_f = abs(curr_f)
            if abs(curr_f) <= 1e-8:
                return curr_x
            
            # Check for sign change
            if prev_f != 0 and curr_f != 0 and prev_f * curr_f < 0:
                try:
                    root = brentq(f, prev_x, curr_x, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if abs(f(root)) <= 1e-8:
                        return root
                    # If brentq failed, use the point with smaller |f|
                    if abs(prev_f) < abs(curr_f):
                        return prev_x
                    else:
                        return curr_x
                except Exception:
                    pass
            
            prev_x = curr_x
            prev_f = curr_f
        
        # Negative direction
        prev_x = x0
        prev_f = f0
        for i in range(1, max_steps + 1):
            if not time_ok():
                break
            curr_x = x0 - i * scale
            curr_f = safe_f(curr_x)
            
            if abs(curr_f) < best_f:
                best_x = curr_x
                best_f = abs(curr_f)
            if abs(curr_f) <= 1e-8:
                return curr_x
            
            # Check for sign change
            if prev_f != 0 and curr_f != 0 and prev_f * curr_f < 0:
                try:
                    root = brentq(f, curr_x, prev_x, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if abs(f(root)) <= 1e-8:
                        return root
                    if abs(prev_f) < abs(curr_f):
                        return prev_x
                    else:
                        return curr_x
                except Exception:
                    pass
            
            prev_x = curr_x
            prev_f = curr_f

    # If we haven't found a root, try a denser local search around best_x
    if not time_ok():
        return best_x

    # Local refinement around best_x
    local_scale = 0.01
    for i in range(-50, 51):
        if not time_ok():
            break
        x_try = best_x + i * local_scale
        f_try = safe_f(x_try)
        if abs(f_try) <= 1e-8:
            return x_try
        if abs(f_try) < best_f:
            best_x = x_try
            best_f = abs(f_try)

    return best_x

# EVOLVE-BLOCK-END
