import numpy as np
from scipy.optimize import brentq
import time

def find_root(f, x0):
    """
    Finds a root of f(x) starting from x0.
    Returns x such that |f(x)| <= 1e-8.
    Uses at most 500 evaluations.
    """
    start_time = time.time()
    deadline = 115.0 # 5s margin from 120s limit
    
    max_evals = 500
    eval_count = 0
    best_x = float(x0)
    best_f = np.inf
    
    def safe_f(x):
        nonlocal eval_count, best_x, best_f
        if eval_count >= max_evals:
            return np.inf
        if time.time() > deadline:
            return np.inf
        try:
            val = f(x)
            eval_count += 1
            if np.isfinite(val):
                if abs(val) < abs(best_f):
                    best_f = val
                    best_x = x
                return float(val)
            else:
                return np.inf
        except Exception:
            eval_count += 1
            return np.inf

    # Initial evaluation
    fx = safe_f(best_x)
    if abs(fx) <= 1e-8:
        return best_x

    # Strategy 1: Find a bracket
    h = 1e-4 * max(1.0, abs(best_x))
    a = None
    b = None
    
    # We keep x0's value for bracketing checks
    x0_val = fx
    
    for i in range(40):
        if time.time() > deadline - 5 or eval_count > max_evals - 10:
            break
            
        x_plus = best_x + h
        x_minus = best_x - h
        
        f_plus = safe_f(x_plus)
        f_minus = safe_f(x_minus)
        
        # Check bracket (best_x, x_plus)
        if np.isfinite(x0_val) and np.isfinite(f_plus) and x0_val * f_plus < 0:
            a, b = best_x, x_plus
            break
            
        # Check bracket (x_minus, best_x)
        if np.isfinite(f_minus) and np.isfinite(x0_val) and f_minus * x0_val < 0:
            a, b = x_minus, best_x
            break
            
        # Check bracket (x_minus, x_plus)
        if np.isfinite(f_minus) and np.isfinite(f_plus) and f_minus * f_plus < 0:
            a, b = x_minus, x_plus
            break
            
        # Expand step
        h *= 2.0

    # If bracket found, use Brent's method
    if a is not None and b is not None:
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if np.isfinite(root):
                f_root = safe_f(root)
                if abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass

    # Strategy 2: Damped Newton fallback
    x_curr = best_x
    f_curr = best_f
    
    if not np.isfinite(f_curr):
        return best_x

    for i in range(20):
        if time.time() > deadline - 2 or eval_count > max_evals - 5:
            break
        if abs(f_curr) <= 1e-8:
            return x_curr
            
        h_d = 1e-6 * max(1.0, abs(x_curr))
        x_probe = x_curr + h_d
        f_probe = safe_f(x_probe)
        
        if not np.isfinite(f_probe):
            break
            
        d = (f_probe - f_curr) / h_d
        
        if abs(d) < 1e-15:
            break
            
        x_new = x_curr - f_curr / d
        
        if not np.isfinite(x_new):
            break
            
        f_new = safe_f(x_new)
        
        if np.isfinite(f_new):
            if abs(f_new) < abs(f_curr):
                x_curr = x_new
                f_curr = f_new
            else:
                break
        else:
            break

    return best_x
