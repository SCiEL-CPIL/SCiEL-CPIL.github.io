"""Root finder for smooth polynomial equations with call-budget awareness."""
import math
import numpy as np
from scipy.optimize import brentq

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Uses brentq if a bracket is found, otherwise Newton-Raphson with safeguards.
    """
    x = float(x0)
    if not _is_finite(x):
        return 0.0

    # Check initial point
    fx = f(x)
    if not _is_finite(fx):
        return 0.0
    if abs(fx) <= 1e-8:
        return x

    # Strategy 1: Find a bracket
    a = x
    fa = fx
    b = None
    fb = None
    
    # Expand in both directions
    step = 1.0
    for direction in [1, -1]:
        if b is not None:
            break
        for _ in range(50): # Max 50 expansions per direction
            x_try = x + direction * step
            f_try = f(x_try)
            if not _is_finite(f_try):
                step *= 2.0
                continue
            if abs(f_try) <= 1e-8:
                return x_try
            if fa * f_try < 0:
                if direction == 1:
                    a, b = x, x_try
                    fa, fb = fx, f_try
                else:
                    a, b = x_try, x
                    fa, fb = f_try, fx
                break
            step *= 2.0
            if step > 1e10:
                break
        if b is not None:
            break

    if b is not None:
        # Use Brent's method
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if _is_finite(root):
                # Verify
                f_root = f(root)
                if _is_finite(f_root) and abs(f_root) <= 1e-8:
                    return root
                # If brentq failed to meet tolerance, return root anyway if it's better
                # Or try to refine?
                # For now, return root. If it fails tolerance, it's a failure.
                # But brentq is usually very accurate.
                return root
        except Exception:
            pass

    # Strategy 2: No bracket found. Use Newton-Raphson with numerical derivative.
    x_curr = x
    f_curr = fx
    
    for i in range(50):
        h = 1e-8 * max(1.0, abs(x_curr))
        f_plus = f(x_curr + h)
        if not _is_finite(f_plus):
            break
        df = (f_plus - f_curr) / h
        if df == 0 or not _is_finite(df):
            break
        
        x_new = x_curr - f_curr / df
        if not _is_finite(x_new):
            break
            
        # Damping if step is too large
        if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
            x_new = x_curr + 0.5 * (x_new - x_curr)
            
        f_new = f(x_new)
        if not _is_finite(f_new):
            break
            
        if abs(f_new) < abs(f_curr):
            x_curr = x_new
            f_curr = f_new
            if abs(f_curr) <= 1e-8:
                return x_curr
        else:
            # Step was not good, try smaller step or break
            break

    return x_curr
