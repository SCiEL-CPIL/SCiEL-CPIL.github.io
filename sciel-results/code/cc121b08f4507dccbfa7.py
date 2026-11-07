"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 120s limit
    x = float(x0)
    
    # Cache to avoid redundant function calls
    cache = {}
    
    def fc(x_val):
        key = x_val
        if key in cache:
            return cache[key]
        val = f(key)
        cache[key] = val
        return val
    
    def elapsed_ok():
        return time.time() < deadline
    
    # Phase 1: Try Newton's method with finite-difference derivative
    # Use a limited number of iterations to avoid diverging
    newton_x = x
    best_x = x
    best_fx = None
    try:
        fx = fc(newton_x)
        if np.isfinite(fx):
            best_x = newton_x
            best_fx = fx
        else:
            fx = 0.0
        if abs(fx) <= 1e-8:
            return newton_x
    except:
        pass
    
    for i in range(15):
        if not elapsed_ok():
            break
        try:
            fx = fc(newton_x)
            if not np.isfinite(fx):
                break
            if abs(fx) <= 1e-12:
                return newton_x
            if abs(fx) < abs(best_fx) if best_fx is not None else True:
                best_x = newton_x
                best_fx = fx
            
            h = 1e-7 * max(1.0, abs(newton_x))
            fx_plus = fc(newton_x + h)
            if not np.isfinite(fx_plus):
                break
            d = (fx_plus - fx) / h
            if not np.isfinite(d) or abs(d) < 1e-15:
                break
            newton_x = newton_x - fx / d
            if not np.isfinite(newton_x):
                break
        except:
            break
    
    # Check if Newton succeeded
    try:
        fx = fc(newton_x)
        if np.isfinite(fx) and abs(fx) <= 1e-8:
            return newton_x
        if np.isfinite(fx) and (best_fx is None or abs(fx) < abs(best_fx)):
            best_x = newton_x
            best_fx = fx
    except:
        pass
    
    # Phase 2: Try to find a bracket using expanding intervals
    # Start from x0 and expand in both directions
    def try_bracket(a, b):
        """Try to find a sign change between a and b."""
        try:
            fa = fc(a)
            fb = fc(b)
            if not np.isfinite(fa) or not np.isfinite(fb):
                return None
            if fa * fb < 0:
                return (a, b)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            return None
        except:
            return None
    
    bracket = None
    # Try small intervals first, then expand
    scale = max(1.0, abs(x))
    for i in range(20):
        if not elapsed_ok():
            break
        delta = scale * (10 ** i)
        if delta > 1e15:
            break
        # Try [x - delta, x + delta]
        result = try_bracket(x - delta, x + delta)
        if result is not None:
            if isinstance(result, tuple):
                bracket = result
            else:
                return result
            break
        # Also try [x, x + delta] and [x - delta, x]
        result = try_bracket(x, x + delta)
        if result is not None:
            if isinstance(result, tuple):
                bracket = result
            else:
                return result
            break
        result = try_bracket(x - delta, x)
        if result is not None:
            if isinstance(result, tuple):
                bracket = result
            else:
                return result
            break
    
    # If we found a bracket, use Brent's method
    if bracket is not None and isinstance(bracket, tuple):
        a, b = bracket
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if np.isfinite(root):
                fx = fc(root)
                if np.isfinite(fx) and abs(fx) <= 1e-8:
                    return root
        except:
            pass
    
    # Phase 3: If still no good root, try golden section search on |f(x)|^2
    # over a reasonable range
    if elapsed_ok():
        # Try to minimize f(x)^2 over an interval around x0
        def objective(x_val):
            try:
                fx = fc(x_val)
                if not np.isfinite(fx):
                    return 1e30
                return fx * fx
            except:
                return 1e30
        
        # Golden section search
        def golden_section_search(func, a, b, tol=1e-12, max_iter=100):
            gr = (math.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc_val = func(c)
            fd_val = func(d)
            for _ in range(max_iter):
                if not elapsed_ok():
                    break
                if b - a < tol:
                    break
                if fc_val < fd_val:
                    b = d
                    d = c
                    fd_val = fc_val
                    c = b - gr * (b - a)
                    fc_val = func(c)
                else:
                    a = c
                    c = d
                    fc_val = fd_val
                    d = a + gr * (b - a)
                    fd_val = func(d)
            return (a + b) / 2
        
        # Try several intervals
        scale = max(1.0, abs(x))
        for i in range(8):
            if not elapsed_ok():
                break
            delta = scale * (10 ** (i + 2))
            if delta > 1e15:
                break
            mid = golden_section_search(objective, x - delta, x + delta)
            if np.isfinite(mid):
                fx = fc(mid)
                if np.isfinite(fx) and abs(fx) <= 1e-8:
                    return mid
                if best_fx is None or abs(fx) < abs(best_fx):
                    best_x = mid
                    best_fx = fx
    
    # Return the best x found
    if best_x is not None and best_fx is not None and np.isfinite(best_fx) and abs(best_fx) <= 1e-8:
        return best_x
    
    # Last resort: return x0 or best_x
    if best_x is not None:
        return best_x
    return x
# EVOLVE-BLOCK-END
