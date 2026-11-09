import math
import time
import sys

# Try to import scipy, fallback to manual implementation if it fails
try:
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def find_root(f, x0):
    """
    Finds a root of f starting near x0.
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = 110.0  # 10s margin before 120s limit
    
    x0 = float(x0)
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)
    
    if best_abs <= 1e-8:
        return best_x
        
    # Bracketing search
    # We look for a sign change in f(x)
    # Start with a small step and expand
    delta = 1e-3
    # We will search in both directions
    # To be efficient, we can check x0, x0+delta, x0-delta
    # If no sign change, expand delta
    
    # Limit the number of expansion steps to prevent infinite loop
    # and to keep call count low.
    # 50 doublings of 1e-3 gives a range of ~1e12, which is huge.
    # 30 doublings gives ~1e6.
    max_expansions = 40
    
    found_bracket = False
    a = None
    b = None
    
    # We already have f(x0)
    fx0 = best_f
    
    for i in range(max_expansions):
        if time.time() - start_time > deadline:
            break
            
        # Check positive direction
        xp = x0 + delta
        fxp = f(xp)
        if abs(fxp) < best_abs:
            best_x = xp
            best_f = fxp
            best_abs = abs(fxp)
            
        # Check negative direction
        xn = x0 - delta
        fxn = f(xn)
        if abs(fxn) < best_abs:
            best_x = xn
            best_f = fxn
            best_abs = abs(fxn)
            
        # Check for sign change between x0 and xp
        if fx0 * fxp < 0:
            a, b = x0, xp
            found_bracket = True
            break
            
        # Check for sign change between x0 and xn
        if fx0 * fxn < 0:
            a, b = xn, x0
            found_bracket = True
            break
            
        # Also check if sign change exists between xp and xn? 
        # That would mean two roots in (xn, xp). 
        # If fxp * fxn < 0, there is a root in (xn, xp).
        # But we need a bracket for brentq.
        # If fxp * fxn < 0, we can bracket (xn, xp).
        if fxp * fxn < 0:
            a, b = xn, xp
            found_bracket = True
            break
            
        # Expand delta
        delta *= 2.0
        
        # Update fx0? No, we are expanding around x0.
        # It might be better to shift the center if we go far, 
        # but for oscillatory functions with likely moderate frequency,
        # expanding from x0 is usually sufficient to find the nearest root.
        
    if found_bracket:
        # Ensure a < b
        if a > b:
            a, b = b, a
            
        try:
            if HAS_SCIPY:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            else:
                # Fallback: Illinois method
                root = illinois(f, a, b)
            
            # Verify the result
            f_root = f(root)
            if abs(f_root) <= 1e-8:
                return root
            else:
                # If brentq failed or returned a bad result, use best_x
                # This can happen if the function is not continuous or behaves weirdly
                pass
                
        except Exception:
            pass

    # If no bracket found or root verification failed, return best_x
    # Check if best_x is good enough
    if abs(best_f) <= 1e-8:
        return best_x
        
    # If not, try a local Newton step from best_x?
    # Or just return best_x. The task says "returns x with |f(x)| <= 1e-8".
    # If we can't find it, returning the best effort is the standard fallback,
    # though it will count as a failure for that equation.
    
    return best_x

def illinois(f, a, b):
    """
    Simple Illinois method implementation for fallback.
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("No sign change")
    
    for _ in range(100):
        # Linear interpolation
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        
        if abs(fc) < 1e-12:
            return c
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            # Illinois modification: halve the weight of the far point
            fa /= 2.0
            
    return (a + b) / 2.0
