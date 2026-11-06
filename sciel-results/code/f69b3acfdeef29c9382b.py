import numpy as np
from scipy.optimize import minimize_scalar, brentq

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses minimization of f(x)^2 to handle ill-conditioned and multiple roots.
    """
    x0 = float(x0)
    
    # Check initial guess
    try:
        fx0 = f(x0)
        if np.isnan(fx0):
            return x0
        if abs(fx0) <= 1e-8:
            return x0
    except Exception:
        return x0

    # Strategy: Minimize g(x) = f(x)^2
    # We need a bounded interval for minimize_scalar with method='bounded'
    # We will expand the interval around x0 until we find a local minimum 
    # or the interval becomes too large.
    
    # Helper to evaluate g(x) = f(x)^2
    def g(x):
        try:
            fx = f(x)
            if np.isnan(fx) or np.isinf(fx):
                return np.inf
            return fx * fx
        except Exception:
            return np.inf

    # Expand window
    # Start with a small relative window
    scale = max(1e-6, abs(x0) * 1e-6)
    for expansion in range(20):
        a = x0 - scale
        b = x0 + scale
        
        # Ensure a and b are finite
        if not (np.isfinite(a) and np.isfinite(b)):
            break
            
        try:
            # Use minimize_scalar with bounded method
            # This is robust and handles multiple roots
            res = minimize_scalar(g, bounds=(a, b), method='bounded', options={'xatol': 1e-12})
            
            if res.success:
                x_root = res.x
                fx_root = f(x_root)
                if abs(fx_root) <= 1e-8:
                    return x_root
                # If the minimum found is not a root, it might be a local minimum of f^2 
                # that is not zero, or the root is outside.
                # We can check if the value at boundaries is smaller, indicating the root is outside.
                # But minimize_scalar on bounded interval finds the global min in the interval.
                # If min(g) > 1e-16 (approx (1e-8)^2), then no root in interval.
                if res.fun > (1e-8)**2:
                    # No root in this interval, expand
                    scale *= 10.0
                    continue
                else:
                    # Found a root candidate
                    return x_root
            else:
                # If optimization failed, try expanding
                scale *= 10.0
        except Exception:
            scale *= 10.0

    # Fallback: If bounded minimization didn't work, try a dense grid search + Newton
    # This is a last resort
    x = x0
    best_x = x0
    best_f = abs(fx0)
    
    # Grid search in a wider range
    grid_scale = max(1.0, abs(x0))
    for i in range(-100, 101):
        xi = x0 + i * grid_scale * 1e-4
        try:
            fxi = f(xi)
            if np.isnan(fxi):
                continue
            if abs(fxi) < best_f:
                best_f = abs(fxi)
                best_x = xi
            if abs(fxi) <= 1e-8:
                return xi
        except Exception:
            pass
    
    # Newton refinement on best_x
    x = best_x
    for _ in range(50):
        try:
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
            h = 1e-8 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0 or np.isnan(d):
                break
            step = fx / d
            # Damping
            if abs(step) > 0.1 * max(1.0, abs(x)):
                step = 0.1 * max(1.0, abs(x)) * np.sign(step)
            x_new = x - step
            if not np.isfinite(x_new):
                break
            x = x_new
        except Exception:
            break
            
    return best_x
