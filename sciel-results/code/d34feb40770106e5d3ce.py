"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    """
    Find a root of f starting near x0 using bounded scalar minimization of f(x)^2.
    Dynamically expands the search interval until a root is found or budget is exhausted.
    """
    x0 = float(x0)
    
    # We'll try expanding intervals around x0
    # Start with a small interval and expand geometrically
    radii = [1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0]
    
    # Also try some offset intervals in case the root is far from x0
    # But first, try centered intervals
    
    best_x = x0
    best_fval = abs(f(x0))
    
    if best_fval <= 1e-8:
        return best_x
    
    for r in radii:
        a = x0 - r
        b = x0 + r
        try:
            res = minimize_scalar(
                lambda x: f(x) ** 2,
                bounds=(a, b),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 100}
            )
            x_candidate = res.x
            if np.isfinite(x_candidate):
                fx_candidate = abs(f(x_candidate))
                if fx_candidate < best_fval:
                    best_x = x_candidate
                    best_fval = fx_candidate
                    if best_fval <= 1e-8:
                        return best_x
        except Exception:
            continue
    
    # If centered intervals didn't work, try some offset intervals
    # The root might be at a specific location relative to x0
    offsets = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 5.0, -5.0, 0.1, -0.1, 10.0, -10.0]
    for off in offsets:
        center = x0 + off
        for r in [0.1, 1.0, 10.0, 100.0]:
            a = center - r
            b = center + r
            try:
                res = minimize_scalar(
                    lambda x: f(x) ** 2,
                    bounds=(a, b),
                    method='bounded',
                    options={'xatol': 1e-10, 'maxiter': 100}
                )
                x_candidate = res.x
                if np.isfinite(x_candidate):
                    fx_candidate = abs(f(x_candidate))
                    if fx_candidate < best_fval:
                        best_x = x_candidate
                        best_fval = fx_candidate
                        if best_fval <= 1e-8:
                            return best_x
            except Exception:
                continue
    
    # Final attempt: use the best x we found
    return best_x
# EVOLVE-BLOCK-END
