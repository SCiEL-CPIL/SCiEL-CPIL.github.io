import math
import time
try:
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def find_root(f, x0):
    start_time = time.time()
    timeout = 145.0
    
    x = float(x0)
    
    try:
        f_x = f(x)
    except:
        return x
        
    if not math.isfinite(f_x):
        return x
        
    if abs(f_x) <= 1e-8:
        return x
        
    best_x = x
    best_f = f_x
    best_abs_f = abs(f_x)
    
    # Phase 1: Bracketing
    a = None
    b = None
    fa = None
    fb = None
    
    step = 1e-4 * max(1.0, abs(x))
    found_bracket = False
    
    for i in range(40):
        if time.time() - start_time > timeout:
            break
            
        current_step = step * (2 ** i)
        
        # Try positive
        x_pos = x + current_step
        try:
            f_pos = f(x_pos)
            if math.isfinite(f_pos):
                if abs(f_pos) < best_abs_f:
                    best_x = x_pos
                    best_f = f_pos
                    best_abs_f = abs(f_pos)
                if f_x * f_pos < 0:
                    a = x
                    b = x_pos
                    fa = f_x
                    fb = f_pos
                    found_bracket = True
                    break
        except:
            pass
            
        # Try negative
        x_neg = x - current_step
        try:
            f_neg = f(x_neg)
            if math.isfinite(f_neg):
                if abs(f_neg) < best_abs_f:
                    best_x = x_neg
                    best_f = f_neg
                    best_abs_f = abs(f_neg)
                if f_x * f_neg < 0:
                    a = x_neg
                    b = x
                    fa = f_neg
                    fb = f_x
                    found_bracket = True
                    break
        except:
            pass
            
        if found_bracket:
            break

    if found_bracket:
        # Use Brent's method if available
        if HAS_SCIPY:
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if math.isfinite(root):
                    try:
                        f_root = f(root)
                        if abs(f_root) <= 1e-8:
                            return root
                    except:
                        pass
            except:
                pass
        
        # Fallback: Pure Python Bisection/Secant Hybrid
        left = a
        right = b
        f_left = fa
        f_right = fb
        
        for _ in range(100):
            if time.time() - start_time > timeout:
                break
                
            mid = (left + right) / 2.0
            try:
                f_mid = f(mid)
            except:
                break
                
            if not math.isfinite(f_mid):
                break
                
            if abs(f_mid) < best_abs_f:
                best_x = mid
                best_f = f_mid
                best_abs_f = abs(f_mid)
                
            if abs(f_mid) <= 1e-8:
                return mid
                
            if f_left * f_mid < 0:
                right = mid
                f_right = f_mid
            else:
                left = mid
                f_left = f_mid
                
            if abs(right - left) < 1e-12 * max(1.0, abs(mid)):
                break
                
        return best_x

    else:
        # No bracket found. Try Secant method blindly.
        x_prev = x
        f_prev = f_x
        h = 1e-6 * max(1.0, abs(x))
        x_curr = x + h
        try:
            f_curr = f(x_curr)
        except:
            return best_x
            
        if not math.isfinite(f_curr):
            return best_x
            
        if abs(f_curr) < best_abs_f:
            best_x = x_curr
            best_f = f_curr
            best_abs_f = abs(f_curr)
            
        for _ in range(50):
            if time.time() - start_time > timeout:
                break
                
            if abs(f_curr) <= 1e-8:
                return x_curr
            if abs(f_prev) <= 1e-8:
                return x_prev
                
            if f_prev == f_curr:
                break
                
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            if not math.isfinite(x_new):
                break
                
            x_prev, f_prev = x_curr, f_curr
            x_curr = x_new
            try:
                f_curr = f(x_curr)
            except:
                break
                
            if not math.isfinite(f_curr):
                break
                
            if abs(f_curr) < best_abs_f:
                best_x = x_curr
                best_f = f_curr
                best_abs_f = abs(f_curr)
                
        return best_x
