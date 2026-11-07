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
    
    # Multi-scale sign-change scan
    # Try scales: 0.1, 0.5, 2.0 (based on cell findings)
    scales = [0.1, 0.5, 2.0]
    
    for scale in scales:
        if not time_ok():
            break
        
        # Scan in both directions from x0
        for direction in [1, -1]:
            if not time_ok():
                break
            
            # Try scanning up to 500 points in this direction/scale
            max_points = 250  # Keep total under 500 calls
            prev_x = x0
            prev_f = f0
            
            for i in range(1, max_points + 1):
                x = x0 + direction * i * scale
                fx = f(x)
                
                if not np.isfinite(fx):
                    continue
                
                if abs(fx) < best_f:
                    best_f = abs(fx)
                    best_x = x
                
                if abs(fx) <= 1e-8:
                    return x
                
                # Check for sign change
                if prev_f * fx < 0:
                    # Found a bracket, use Brent's method
                    try:
                        root = brentq(f, prev_x, x, xtol=1e-12, rtol=1e-12, maxiter=100)
                        if np.isfinite(root):
                            return root
                    except Exception:
                        pass
                
                prev_x = x
                prev_f = fx
    
    # If we haven't found a root yet, try Newton's method as fallback
    x = float(best_x)
    for _ in range(50):
        if not time_ok():
            break
        fx = f(x)
        if not np.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        h = 1e-6 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not np.isfinite(d):
            break
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
        x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
