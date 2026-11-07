import math
import time
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a hybrid strategy:
    1. Check if x0 is already a root.
    2. Try to find a bracket [a, b] where f(a) and f(b) have opposite signs.
       If found, use scipy.optimize.brentq which is robust and efficient.
    3. If no bracket is found, use a safeguarded Newton's method with finite difference derivative.
    Includes a deadline guard to prevent timeout.
    """
    TOL = 1e-8
    DEADLINE = 140.0  # Safety margin before 150s kill
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0 # Should not happen for valid f, but safe
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    # Strategy 1: Bracketing + Brent's Method
    # Try to find a bracket by expanding from x0
    a = x0 - 1.0
    b = x0 + 1.0
    fa = None
    fb = None
    
    # We need to evaluate f(a) and f(b)
    # To save calls, we can reuse fx0 if we start bracket around 0? 
    # No, x0 is arbitrary.
    
    bracket_found = False
    # Expand bracket
    for i in range(30): # Max expansion steps
        if not check_deadline():
            break
            
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            break
            
        if not (is_finite(fa) and is_finite(fb)):
            break
            
        if fa == 0:
            return a
        if fb == 0:
            return b
            
        if fa * fb < 0:
            bracket_found = True
            break
            
        # Expand
        a = x0 - (x0 - a) * 2
        b = x0 + (b - x0) * 2
        
        # Safety: don't expand to infinity
        if abs(a) > 1e10 or abs(b) > 1e10:
            break

    if bracket_found:
        try:
            # brentq is very robust
            root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if is_finite(root):
                return root
        except Exception:
            pass # Fall through to Newton

    # Strategy 2: Safeguarded Newton's Method
    x = float(x0)
    fx = fx0
    
    for i in range(60):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
            
        if not is_finite(fx_plus):
            break
            
        dfx = (fx_plus - fx) / h
        
        if dfx == 0:
            # Derivative is zero, try a small step
            x_new = x + 1e-4
            fx = f(x_new)
            if is_finite(fx):
                x = x_new
            else:
                break
            continue
            
        x_new = x - fx / dfx
        
        # Safeguard: if step is too large or non-finite, dampen
        if not is_finite(x_new) or abs(x_new - x) > 1e6 * max(1.0, abs(x)):
            x_new = x - 0.5 * (fx / dfx)
            
        if not is_finite(x_new):
            break
            
        # Evaluate new point
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not is_finite(fx_new):
            # Dampen further
            x_new = x - 0.1 * (fx / dfx)
            if not is_finite(x_new):
                break
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not is_finite(fx_new):
                break
                
        x = x_new
        fx = fx_new
        
    return x
# EVOLVE-BLOCK-END
