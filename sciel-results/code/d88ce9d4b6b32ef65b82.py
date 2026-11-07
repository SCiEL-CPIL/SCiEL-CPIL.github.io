import scipy.optimize as so
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Check initial point
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # 2. If we are close, try Newton's method (fast quadratic convergence)
    #    Use central difference for derivative
    if abs(fx0) < 1e-4:
        x = x0
        fx = fx0
        for _ in range(10):
            if time.time() > deadline:
                return x
            if abs(fx) <= 1e-8:
                return x
            
            h = 1e-8 * max(1.0, abs(x))
            # Central difference
            f_plus = f(x + h)
            f_minus = f(x - h)
            dfx = (f_plus - f_minus) / (2 * h)
            
            if dfx == 0:
                break
                
            x_new = x - fx / dfx
            
            # Safeguard against divergence
            if not (-1e308 < x_new < 1e308):
                break
                
            fx_new = f(x_new)
            
            # If it got worse significantly, stop Newton and try bracketing
            if abs(fx_new) > 10 * abs(fx):
                break
                
            x = x_new
            fx = fx_new
            
        if abs(fx) <= 1e-8:
            return x

    # 3. Try to find a bracket around x0
    #    Start with a reasonable step size
    delta = 1.0
    a = x0 - delta
    b = x0 + delta
    
    # Check if the root is in [a, b]
    fa = f(a)
    fb = f(b)
    
    # Helper to check for sign change
    def has_sign_change(f1, f2):
        return (f1 < 0 and f2 > 0) or (f1 > 0 and f2 < 0)

    if has_sign_change(fa, fb):
        try:
            root = so.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            return float(root)
        except Exception:
            pass

    # Expand bracket if not found
    for _ in range(10):
        if time.time() > deadline:
            break
        delta *= 2.0
        a = x0 - delta
        b = x0 + delta
        fa = f(a)
        fb = f(b)
        
        if has_sign_change(fa, fb):
            try:
                root = so.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                return float(root)
            except Exception:
                break

    # 4. Fallback: Secant Method
    #    Use x0 and x0+1 as initial points
    x_prev = x0
    x_curr = x0 + 1.0
    f_prev = fx0
    f_curr = f(x_curr)
    
    best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
    best_f = min(abs(f_prev), abs(f_curr))
    
    for _ in range(50):
        if time.time() > deadline:
            break
            
        if abs(f_curr) <= 1e-8:
            return x_curr
            
        if abs(f_curr) < best_f:
            best_f = abs(f_curr)
            best_x = x_curr
            
        if f_curr == f_prev:
            # Stuck, nudge
            x_curr += 1e-6
            f_curr = f(x_curr)
            continue
            
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # Safeguard
        if not (-1e308 < x_new < 1e308):
            break
            
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f(x_curr)
        
        if abs(f_curr) < best_f:
            best_f = abs(f_curr)
            best_x = x_curr

    return best_x
# EVOLVE-BLOCK-END
