"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin before 150s kill
    x0 = float(x0)
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT
    
    # Safely evaluate f
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
    
    def update_best(x, fx):
        nonlocal best_x, best_f
        if abs(fx) < best_f:
            best_x = x
            best_f = abs(fx)
            if best_f <= 1e-8:
                return True
        return False
    
    # Phase 1: Multi-scale sign-change scan to find brackets
    # Try several scales to find where f changes sign
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    brackets = []  # list of (a, b, fa, fb) where fa*fb < 0
    
    for scale in scales:
        if not time_ok():
            break
        
        # Scan in both directions from x0 with this scale
        for direction in [1, -1]:
            for i in range(1, 20):
                if not time_ok():
                    break
                
                x1 = x0 + direction * i * scale
                x2 = x0 + direction * (i + 1) * scale
                f1 = safe_f(x1)
                f2 = safe_f(x2)
                
                # Update best
                if update_best(x1, f1):
                    return x1
                if update_best(x2, f2):
                    return x2
                
                # Check for sign change
                if np.isfinite(f1) and np.isfinite(f2) and f1 * f2 < 0:
                    brackets.append((x1, x2, f1, f2))
                    if len(brackets) >= 5:
                        break
            
            if len(brackets) >= 5:
                break
        
        if len(brackets) >= 5:
            break
    
    # Phase 2: If we found brackets, use brentq on the tightest ones
    if brackets:
        # Sort brackets by width (tightest first)
        brackets.sort(key=lambda b: abs(b[1] - b[0]))
        
        for a, b, fa, fb in brackets[:5]:
            if not time_ok():
                break
            
            try:
                # Try brentq with tight tolerance
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if np.isfinite(root):
                    froot = safe_f(root)
                    if abs(froot) <= 1e-8:
                        return root
                    update_best(root, froot)
            except Exception:
                pass
    
    # Phase 3: If no brackets or brentq failed, try Newton's method with fallback
    x = x0
    for _ in range(50):
        if not time_ok():
            break
        
        fx = safe_f(x)
        if not np.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        update_best(x, fx)
        
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        fx_plus = safe_f(x + h)
        if not np.isfinite(fx_plus):
            break
        d = (fx_plus - fx) / h
        if d == 0 or not np.isfinite(d):
            break
        
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
        
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
        
        x = x_new
    
    # Phase 4: If still not converged, try a finer local scan around best_x
    if best_f > 1e-8 and time_ok():
        # Scan around best_x with fine steps
        step = 0.01 * max(1.0, abs(best_x))
        for i in range(-50, 51):
            if not time_ok():
                break
            xi = best_x + i * step
            fi = safe_f(xi)
            if update_best(xi, fi):
                return xi
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
