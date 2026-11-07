import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    x0 = float(x0)
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0
        
    # Try to find a bracket
    step = 1.0
    a = x0 - step
    b = x0 + step
    fa = f(a)
    fb = f(b)
    
    # If we already have a sign change, great
    if fa * fb > 0:
        # Expand
        for _ in range(20):
            step *= 2.0
            a = x0 - step
            b = x0 + step
            fa = f(a)
            fb = f(b)
            if fa * fb <= 0:
                break
            # If step becomes too large, stop
            if step > 1e9:
                break
    
    if fa * fb <= 0:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12)
            return root
        except Exception:
            pass
            
    # Fallback: Minimize f(x)^2 in the last bracket [a, b]
    # This handles even multiplicity roots or if bracketing failed
    try:
        res = minimize_scalar(lambda x: f(x)**2, bounds=(a, b), method='bounded')
        return res.x
    except Exception:
        return x0
