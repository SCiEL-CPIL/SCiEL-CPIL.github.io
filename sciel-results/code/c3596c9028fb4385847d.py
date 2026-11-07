import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    
    # Check initial point
    f0 = f(x0)
    if abs(f0) <= 1e-8:
        return x0
    
    def try_bracket(a, b, fa, fb):
        if fa * fb < 0:
            try:
                # brentq requires fa and fb to have different signs
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                # Verify
                fr = f(root)
                if abs(fr) <= 1e-8:
                    return root
            except Exception:
                pass
        return None

    # Multi-scale scanning strategy
    # Scales: (step_size, range_radius)
    # We scan right and left for each scale.
    # Total calls estimate: sum(2 * (2 * range / step))
    # We want to stay under ~450 calls to be safe.
    
    scales = [
        (0.01, 0.5),   # 100 calls total (50 right, 50 left)
        (0.05, 2.5),   # 100 calls total (50 right, 50 left)
        (0.25, 12.5),  # 100 calls total (50 right, 50 left)
        (1.25, 62.5),  # 100 calls total (50 right, 50 left)
    ]
    
    last_x = x0
    last_f = f0
    
    for step, radius in scales:
        # Scan Right
        last_x = x0
        last_f = f0
        for i in range(1, int(radius / step) + 1):
            next_x = x0 + i * step
            fn = f(next_x)
            if last_f * fn < 0:
                res = try_bracket(last_x, next_x, last_f, fn)
                if res is not None:
                    return res
            last_x = next_x
            last_f = fn
            
        # Scan Left
        last_x = x0
        last_f = f0
        for i in range(1, int(radius / step) + 1):
            next_x = x0 - i * step
            fn = f(next_x)
            if last_f * fn < 0:
                res = try_bracket(next_x, last_x, fn, last_f)
                if res is not None:
                    return res
            last_x = next_x
            last_f = fn
            
    # Fallback: If no root found, return x0
    return x0
