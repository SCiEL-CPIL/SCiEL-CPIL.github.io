import numpy as np
import time
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def check_deadline():
        return time.time() > deadline

    x0 = float(x0)
    
    # 1. Check if x0 is already a root
    fx0 = f(x0)
    if not np.isfinite(fx0):
        # If f(x0) is bad, try a small perturbation
        x0 = x0 + 1e-6
        fx0 = f(x0)
    
    if abs(fx0) <= 1e-8:
        return x0

    # 2. Try Newton's Method
    # Estimate derivative at x0
    h = 1e-5 * max(1.0, abs(x0))
    try:
        fxp = f(x0 + h)
        fxm = f(x0 - h)
        if np.isfinite(fxp) and np.isfinite(fxm):
            dfx0 = (fxp - fxm) / (2 * h)
        else:
            dfx0 = None
    except Exception:
        dfx0 = None

    x = x0
    fx = fx0
    dfx = dfx0

    newton_success = False
    for i in range(15):
        if check_deadline():
            break
            
        if dfx is None or abs(dfx) < 1e-12:
            # Derivative is zero or unknown, try to re-estimate or break
            if i == 0:
                # Try a different h
                h = 1e-4 * max(1.0, abs(x))
                try:
                    fxp = f(x + h)
                    fxm = f(x - h)
                    if np.isfinite(fxp) and np.isfinite(fxm):
                        dfx = (fxp - fxm) / (2 * h)
                    else:
                        break
                except Exception:
                    break
            else:
                break
        
        x_new = x - fx / dfx
        
        if not np.isfinite(x_new):
            break
            
        # Check for convergence
        if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
            fx_new = f(x_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            break

        x = x_new
        fx = f(x)
        
        if not np.isfinite(fx):
            break
            
        if abs(fx) <= 1e-8:
            return x

        # Update derivative
        h = 1e-5 * max(1.0, abs(x))
        try:
            fxp = f(x + h)
            fxm = f(x - h)
            if np.isfinite(fxp) and np.isfinite(fxm):
                dfx = (fxp - fxm) / (2 * h)
            else:
                dfx = None
        except Exception:
            dfx = None

        if dfx is None:
            break

    # 3. Fallback: Find a bracket and use Brent's Method
    # Expand from x0 to find a sign change
    a = x0
    fa = f(a)
    
    # If Newton got us closer, use that as center? 
    # Let's use the best x found or x0.
    # Actually, just expand from x0 is safer for bracketing.
    
    # Try expanding intervals
    # We want f(a) and f(b) to have opposite signs.
    # Start with small radius, expand exponentially.
    
    # First, try to see if the current x from Newton is in a good spot?
    # No, bracketing needs a sign change.
    
    radius = 1.0
    max_radius = 1e6
    
    found_bracket = False
    b = None
    fb = None
    
    # Check if x0 itself is part of a bracket with a nearby point?
    # We already know f(x0).
    
    for _ in range(20):
        if check_deadline():
            break
            
        # Try [x0 - radius, x0 + radius]
        # We already have f(x0). We need a point with opposite sign.
        
        # Try left side
        a_left = x0 - radius
        try:
            fa_left = f(a_left)
            if np.isfinite(fa_left):
                if fa_left * fx0 < 0:
                    a = a_left
                    fa = fa_left
                    b = x0
                    fb = fx0
                    found_bracket = True
                    break
        except Exception:
            pass
            
        # Try right side
        b_right = x0 + radius
        try:
            fb_right = f(b_right)
            if np.isfinite(fb_right):
                if fb_right * fx0 < 0:
                    a = x0
                    fa = fx0
                    b = b_right
                    fb = fb_right
                    found_bracket = True
                    break
        except Exception:
            pass
        
        radius *= 2.0
        if radius > max_radius:
            break

    if found_bracket:
        try:
            root = brentq(f, a, b, xtol=1e-12, maxiter=50)
            if np.isfinite(root):
                return root
        except Exception:
            pass

    # 4. Final fallback: Return best x found
    # If we didn't find a bracket, return x0 or the last Newton x?
    # The problem says "returns x with |f(x)| <= 1e-8".
    # If we fail, we return the best we have.
    
    # Let's try one more thing: Secant method from x0 and x0+h
    # This doesn't require derivative.
    
    x1 = x0
    x2 = x0 + 1e-5
    f1 = f(x1)
    f2 = f(x2)
    
    if np.isfinite(f1) and np.isfinite(f2):
        for i in range(20):
            if check_deadline():
                break
            if abs(f2) < abs(f1):
                x1, x2 = x2, x1
                f1, f2 = f2, f1
            
            if abs(f2) <= 1e-8:
                return x2
                
            if abs(x2 - x1) < 1e-14 * max(1.0, abs(x2)):
                break
                
            # Secant step
            denom = f2 - f1
            if abs(denom) < 1e-15:
                break
            x3 = x2 - f2 * (x2 - x1) / denom
            if not np.isfinite(x3):
                break
            f3 = f(x3)
            if not np.isfinite(f3):
                break
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            
        if abs(f2) <= 1e-8:
            return x2
        # Return the best of x1, x2
        if abs(f1) < abs(f2):
            return x1
        else:
            return x2

    return x0
