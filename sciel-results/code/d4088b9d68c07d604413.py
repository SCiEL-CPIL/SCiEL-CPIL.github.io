import time
import numpy as np
from scipy.optimize import brentq

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
    # Try scales: 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000
    scales = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    bracket_found = False
    a, b, fa, fb = None, None, None, None
    
    for scale in scales:
        if not time_ok():
            break
        # Scan outward in both directions
        # Step size within this scale region
        n_steps = 20  # points per side
        for direction in [1, -1]:
            for i in range(1, n_steps + 1):
                if not time_ok():
                    break
                x = x0 + direction * i * scale
                fx = f(x)
                if not np.isfinite(fx):
                    continue
                if abs(fx) < best_f:
                    best_f = abs(fx)
                    best_x = x
                if abs(fx) <= 1e-8:
                    return x
                
                # Check for sign change with previous point
                if i == 1:
                    prev_x = x0
                    prev_f = f0
                else:
                    prev_x = x0 + direction * (i - 1) * scale
                    prev_f = None  # We didn't store it, so just check consecutive
                
                # We need to track previous f value
                # Let's restructure: keep track of last point
                # Actually, let's do a simpler approach: scan and check consecutive pairs
    
    # Better approach: do a structured scan
    # Phase 1: Find a bracket by scanning outward
    a, b, fa, fb = None, None, None, None
    
    # Scan with adaptive step sizes
    # Start with small steps near x0, then expand
    scan_points = []
    scan_fvals = []
    
    # Generate scan points: logarithmic spacing outward
    # Near: x0 + i*delta for i in 1..20, delta = 1e-3
    # Far: x0 + i*delta for i in 1..20, delta = 0.1
    # Very far: x0 + i*delta for i in 1..10, delta = 10.0
    
    deltas = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    steps_per_delta = [20, 20, 20, 10, 10, 5]
    
    prev_x = x0
    prev_f = f0
    bracket_found = False
    
    for delta, n_steps in zip(deltas, steps_per_delta):
        if not time_ok():
            break
        for direction in [1, -1]:
            if not time_ok():
                break
            for i in range(1, n_steps + 1):
                if not time_ok():
                    break
                x = x0 + direction * i * delta
                fx = f(x)
                if not np.isfinite(fx):
                    continue
                if abs(fx) < best_f:
                    best_f = abs(fx)
                    best_x = x
                if abs(fx) <= 1e-8:
                    return x
                
                # Check sign change between prev and current
                if np.isfinite(prev_f) and prev_f * fx < 0:
                    a, b = prev_x, x
                    fa, fb = prev_f, fx
                    bracket_found = True
                    break
                
                prev_x, prev_f = x, fx
            if bracket_found:
                break
        if bracket_found:
            break
    
    # If bracket found, use Brent's method
    if bracket_found and time_ok():
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if np.isfinite(root):
                return root
        except Exception:
            pass
    
    # If no bracket found, try to minimize |f| using golden section search
    # in a reasonable range around x0
    if time_ok():
        # Try golden section on |f| in [x0-100, x0+100]
        try:
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(lambda x: abs(f(x)), 
                                     bounds=(x0 - 100, x0 + 100), 
                                     method='bounded',
                                     options={'xatol': 1e-10, 'maxiter': 50})
            if result.success and np.isfinite(result.x):
                x_min = result.x
                f_min = f(x_min)
                if np.isfinite(f_min) and abs(f_min) < best_f:
                    best_x = x_min
                    best_f = abs(f_min)
                if abs(f_min) <= 1e-8:
                    return x_min
        except Exception:
            pass
    
    # Final fallback: return best point found
    return best_x
