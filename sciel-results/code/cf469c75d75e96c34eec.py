import time
import math
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root x of f(x) = 0 such that |f(x)| <= 1e-8.
    Uses a hybrid strategy: Secant method for fast local convergence,
    geometric bracketing to find an interval, and Brent's method for
    robust and efficient final convergence.
    """
    start_time = time.time()
    DEADLINE = 10.0  # Safety margin well below 120s/150s limits
    
    x = float(x0)
    best_x = x
    best_f = None
    try:
        best_f = f(x)
    except Exception:
        pass

    if best_f is not None and abs(best_f) <= 1e-8:
        return x

    # Phase 1: Secant Method (Fast, 1 new call per step)
    # We need two points to start.
    # If x0 is 0, step is 1e-4, else relative step.
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    x0_sec = x
    x1_sec = x1
    
    f0 = best_f
    try:
        f1 = f(x1_sec)
    except Exception:
        f1 = None

    if f1 is None or not np.isfinite(f1):
        # Fallback to bracketing immediately if first step fails
        bracket = find_bracket(f, x, f0)
        if bracket:
            return solve_brent(f, bracket[0], bracket[1], best_x, best_f, start_time, DEADLINE)
        return best_x

    # Check sign change in initial step
    if f0 * f1 < 0:
        return solve_brent(f, x0_sec, x1_sec, best_x, best_f, start_time, DEADLINE)
        
    if abs(f1) <= 1e-8:
        return x1_sec

    # Iterate Secant
    for _ in range(15):
        if time.time() - start_time > DEADLINE:
            return best_x
            
        if f1 == 0:
            return x1_sec
        if f0 == f1:
            break
            
        # Secant step
        dx = (f1 - f0) / (x1_sec - x0_sec)
        if dx == 0:
            break
        x_new = x1_sec - f1 / dx
        
        # Check for divergence
        if not np.isfinite(x_new):
            break
            
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if not np.isfinite(f_new):
            break
            
        # Update best
        if best_f is None or abs(f_new) < abs(best_f):
            best_x = x_new
            best_f = f_new
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Check for sign change
        if f1 * f_new < 0:
            return solve_brent(f, x1_sec, x_new, best_x, best_f, start_time, DEADLINE)
            
        # Shift points
        x0_sec = x1_sec
        f0 = f1
        x1_sec = x_new
        f1 = f_new
        
        # If step is very small, we might be stuck or converged
        if abs(x_new - x1_sec) < 1e-12 * max(1.0, abs(x_new)):
            break

    # Phase 2: Bracketing
    # Use the best point found so far as center
    center_x = best_x
    center_f = best_f
    
    bracket = find_bracket(f, center_x, center_f)
    
    if bracket:
        return solve_brent(f, bracket[0], bracket[1], best_x, best_f, start_time, DEADLINE)
        
    # If no bracket found, return best attempt
    return best_x

def find_bracket(f, x_center, f_center):
    """
    Expands a bracket [a, b] around x_center until a sign change is found.
    Returns (a, b) or None.
    """
    if f_center is None:
        return None
        
    # Initial radius
    r = 1e-3 * max(1.0, abs(x_center))
    a = x_center - r
    b = x_center + r
    
    fa = None
    fb = None
    
    try:
        fa = f(a)
    except Exception:
        pass
    try:
        fb = f(b)
    except Exception:
        pass
        
    # Check initial bracket
    if fa is not None and fb is not None:
        if fa * fb < 0:
            return (a, b)
        if abs(fa) <= 1e-8:
            return (a, b) # Degenerate case, but brentq handles it? No, brentq needs sign change or 0.
            # Actually if fa is 0, we can just return a. But let's let brentq handle it or check outside.
            # To be safe, if we find a zero, we can return a dummy bracket or handle it.
            # The caller solve_brent will check.
            # Let's return a bracket where one end is the root? No, brentq requires f(a)*f(b)<=0.
            # If fa==0, we can return (a, b) and brentq will return a.
    
    # Expand geometrically
    for _ in range(50):
        r *= 2.0
        a = x_center - r
        b = x_center + r
        
        try:
            fa = f(a)
        except Exception:
            fa = None
        try:
            fb = f(b)
        except Exception:
            fb = None
            
        if fa is None or fb is None:
            continue
            
        if fa * fb < 0:
            return (a, b)
            
        # If we go too far, stop
        if r > 1e10:
            break
            
    return None

def solve_brent(f, a, b, best_x, best_f, start_time, DEADLINE):
    """
    Uses scipy.optimize.brentq to find the root in [a, b].
    """
    if time.time() - start_time > DEADLINE:
        return best_x
        
    try:
        root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
        f_root = f(root)
        if abs(f_root) <= 1e-8:
            return root
        else:
            # Brent failed to meet tolerance? Unlikely, but check best.
            if best_f is None or abs(f_root) < abs(best_f):
                return root
            return best_x
    except Exception:
        # If brentq fails, return best known
        return best_x

# EVOLVE-BLOCK-END
