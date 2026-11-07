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
    
    def try_bracket(a, b):
        """Try to find a root in [a, b] using brentq."""
        nonlocal best_x, best_f
        try:
            fa = f(a)
            fb = f(b)
            if not np.isfinite(fa) or not np.isfinite(fb):
                return None
            if abs(fa) <= 1e-8:
                best_x = a
                best_f = abs(fa)
                return a
            if abs(fb) <= 1e-8:
                best_x = b
                best_f = abs(fb)
                return b
            if fa * fb < 0:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if np.isfinite(root):
                    return root
        except Exception:
            pass
        return None
    
    # Phase 1: Geometric expansion to find a bracket
    # This is efficient for finding brackets over large ranges
    h = max(1.0, 0.1 * abs(x0))
    
    prev_x_pos = x0
    prev_f_pos = f0
    prev_x_neg = x0
    prev_f_neg = f0
    
    bracket_found = False
    a = 0.0
    b = 0.0
    
    # Expand up to 25 steps
    for k in range(25):
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
            
            if prev_f_neg * f_neg < 0:
                a, b = x_neg, prev_x_neg
                bracket_found = True
                break
            
            prev_x_neg = x_neg
            prev_f_neg = f_neg
    
    if bracket_found:
        root = try_bracket(a, b)
        if root is not None:
            return root
    
    # Phase 2: If no bracket found, try multi-scale linear scans
    # This is better for oscillatory functions where geometric expansion
    # might skip over sign changes
    scales = [0.1, 0.5, 2.0]
    
    for scale in scales:
        if not time_ok():
            break
        
        for direction in [1, -1]:
            if not time_ok():
                break
            
            max_points = 100  # Limit to save calls for other phases
            prev_x = x0
            prev_f = f0
            
            for i in range(1, max_points + 1):
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
                
                if prev_f * fx < 0:
                    root = try_bracket(prev_x, x)
                    if root is not None:
                        return root
                
                prev_x = x
                prev_f = fx
    
    # Phase 3: Newton's method from best_x found so far
    # This can quickly converge if we're near a root
    x = float(best_x)
    for _ in range(30):
        if not time_ok():
            break
        
        fx = f(x)
        if not np.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        
        # Central difference for better accuracy
        h = 1e-6 * max(1.0, abs(x))
        fx_plus = f(x + h)
        fx_minus = f(x - h)
        
        if not np.isfinite(fx_plus) or not np.isfinite(fx_minus):
            break
        
        d = (fx_plus - fx_minus) / (2 * h)
        if d == 0 or not np.isfinite(d):
            break
        
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
        
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + np.sign(x_new - x) * 10 * max(1.0, abs(x))
        
        x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
