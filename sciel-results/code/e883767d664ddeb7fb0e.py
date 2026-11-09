"""Root finder for 3 fixed oscillatory equations."""
import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Try Newton's method first (fast convergence when close to root).
      2. If Newton fails or diverges, find a bracket via expansion.
      3. Use Brent's method on the bracket for guaranteed convergence.
      4. Fallback to secant method if no bracket found.
    """
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    call_count = 0
    max_calls = 480  # Leave buffer for safety
    
    def safe_call(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise Exception("Call limit exceeded")
        call_count += 1
        return f(x)
    
    def time_ok():
        return time.time() < deadline

    x0 = float(x0)
    best_x = x0
    best_f = None
    
    try:
        best_f = safe_call(best_x)
    except Exception:
        return best_x
    
    if not math.isfinite(best_f):
        return best_x
        
    if abs(best_f) <= 1e-8:
        return best_x

    # Strategy 1: Newton's method with sign-change detection
    # This is fast when we're close to a root
    x = x0
    fx = best_f
    history = [(x, fx)]
    
    for i in range(15):
        if not time_ok():
            break
            
        # Check for sign change in recent history
        if len(history) >= 2:
            prev_x, prev_fx = history[-2]
            curr_x, curr_fx = history[-1]
            if prev_fx * curr_fx < 0:
                # Bracket found in Newton steps
                try:
                    root = brentq(safe_call, prev_x, curr_x, xtol=1e-12, rtol=1e-12, maxiter=50)
                    if math.isfinite(root):
                        return root
                except Exception:
                    pass
        
        # Compute derivative via finite difference
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = safe_call(x + h)
        except Exception:
            break
            
        if not math.isfinite(fx_plus):
            break
            
        d = (fx_plus - fx) / h
        
        if not math.isfinite(d) or abs(d) < 1e-15:
            break
            
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
        
        # Check if step is reasonable
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            break
            
        try:
            fx_new = safe_call(x_new)
        except Exception:
            break
        
        if not math.isfinite(fx_new):
            break
        
        if abs(fx_new) < abs(best_f):
            best_x = x_new
            best_f = fx_new
            
        if abs(fx_new) <= 1e-8:
            return x_new
            
        history.append((x_new, fx_new))
        x = x_new
        fx = fx_new
        
        # If getting worse significantly, stop Newton
        if i > 3 and abs(fx_new) > 2 * abs(fx):
            break

    # Strategy 2: Find a bracket by expanding from best_x
    center = best_x
    found_bracket = False
    a = None
    b = None
    fa = None
    fb = None
    
    # Try to find a bracket by expanding from center
    # Check points in increasing distance from center
    distances = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 
                 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0, 500000.0]
    
    last_r_x = center
    last_r_fx = best_f
    last_l_x = center
    last_l_fx = best_f

    for d in distances:
        if not time_ok():
            break
        
        # Check right
        x_r = center + d
        try:
            f_r = safe_call(x_r)
        except Exception:
            break
        if math.isfinite(f_r):
            if abs(f_r) < abs(best_f):
                best_x = x_r
                best_f = f_r
            if abs(f_r) <= 1e-8:
                return x_r
            if last_r_fx * f_r < 0:
                a = last_r_x
                fa = last_r_fx
                b = x_r
                fb = f_r
                found_bracket = True
                break
            last_r_x = x_r
            last_r_fx = f_r

        # Check left
        x_l = center - d
        try:
            f_l = safe_call(x_l)
        except Exception:
            break
        if math.isfinite(f_l):
            if abs(f_l) < abs(best_f):
                best_x = x_l
                best_f = f_l
            if abs(f_l) <= 1e-8:
                return x_l
            if last_l_fx * f_l < 0:
                a = x_l
                fa = f_l
                b = last_l_x
                fb = last_l_fx
                found_bracket = True
                break
            last_l_x = x_l
            last_l_fx = f_l

    if found_bracket:
        try:
            # Use Brent's method
            root = brentq(safe_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass

    # Strategy 3: Fallback - Secant method
    x1 = x0
    fx1 = best_f
    x2 = x0 + 1e-4 * max(1.0, abs(x0))
    try:
        fx2 = safe_call(x2)
    except Exception:
        return best_x
    
    if not math.isfinite(fx2):
        return best_x

    # Secant iteration
    for i in range(80):
        if not time_ok():
            break
        if abs(fx2) <= 1e-8:
            return x2
        if abs(fx1) <= 1e-8:
            return x1
            
        if fx1 == fx2:
            break
            
        x3 = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
        
        if not math.isfinite(x3):
            break
            
        x1, x2 = x2, x3
        fx1, fx2 = fx2, safe_call(x2)
        
        if abs(fx2) < abs(best_f):
            best_x = x2
            best_f = fx2

    return best_x
# EVOLVE-BLOCK-END
