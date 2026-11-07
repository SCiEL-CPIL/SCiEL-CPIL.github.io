"""
Strategy: Global Discrete Minimization using scipy.optimize.minimize_scalar.

The current approach relies on local methods (Newton, Secant) and manual bracketing.
This new approach reframes the problem as minimizing the scalar function g(x) = |f(x)|.
We use scipy's bounded minimization (Brent's method with bounds) over a wide,
pre-defined window. This is robust to oscillatory behavior because it doesn't
rely on derivative signs or local convergence basins, but rather on finding the
global minimum of the magnitude within the search space.
"""

import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

# Global deadline: evaluator kills after 150s, we use 145s as a safety margin
_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 480  # Stay well under 500 per equation

def _time_left():
    return _DEADLINE - time.time()

def _counted_call(f, x, counter):
    """Call f(x) and track the call count. Returns (result, should_stop)."""
    counter[0] += 1
    if counter[0] >= _CALL_BUDGET:
        # Return a neutral value to stop the optimizer, but don't crash
        return 1e10, True
    try:
        val = f(x)
    except Exception:
        return 1e10, True
    if not math.isfinite(val):
        return 1e10, True
    return val, False

def _minimize_scalar_wrapper(f, counter, bounds):
    """
    Wraps the objective function for scipy.optimize.minimize_scalar.
    The objective is |f(x)|.
    """
    def objective(x):
        if _time_left() < 1.0:
            return 1e10
        val, stop = _counted_call(f, x, counter)
        if stop:
            return 1e10
        return abs(val)
    return objective

def find_root(f, x0):
    """
    Find a root x of f(x) = 0 such that |f(x)| <= 1e-8.
    Uses global minimization of |f(x)| over a wide bounded interval.
    """
    counter = [0]
    x0 = float(x0)
    
    # Define search windows. 
    # The problem states "oscillatory equations". 
    # We assume the root is likely near x0 or within a reasonable range.
    # We try multiple windows of increasing size, but bounded to avoid
    # wasting calls on irrelevant regions.
    
    # Initial window centered on x0
    # We use a logarithmic scale of widths to cover different magnitudes
    widths = [1.0, 5.0, 20.0, 100.0, 1000.0]
    
    best_x = x0
    best_val = float('inf')
    
    # First, evaluate at x0 to have a baseline
    val0, stop = _counted_call(f, x0, counter)
    if not stop and math.isfinite(val0):
        best_x = x0
        best_val = abs(val0)
        if best_val <= 1e-8:
            return best_x

    # Try bounded minimization in different windows
    for width in widths:
        if _time_left() < 2.0 or counter[0] > _CALL_BUDGET - 20:
            break
            
        a = x0 - width
        b = x0 + width
        
        # Skip if window is too small or invalid
        if b <= a:
            continue
            
        try:
            objective = _minimize_scalar_wrapper(f, counter, (a, b))
            # Use 'bounded' method which uses Brent's algorithm with bounds
            # maxiter limits the number of iterations (function evaluations)
            # We set maxiter conservatively to stay within budget
            res = minimize_scalar(
                objective, 
                bounds=(a, b), 
                method='bounded', 
                options={'maxiter': 50, 'xatol': 1e-10}
            )
            
            if res.success:
                x_opt = res.x
                # Verify the result
                f_opt, stop = _counted_call(f, x_opt, counter)
                if not stop and math.isfinite(f_opt):
                    val_opt = abs(f_opt)
                    if val_opt < best_val:
                        best_x = x_opt
                        best_val = val_opt
                        if best_val <= 1e-8:
                            return best_x
                            
        except Exception:
            # If minimization fails, continue to next window
            continue
            
    # If we haven't found a root yet, try a few local Newton steps from best_x
    # as a final refinement, since minimization might stop at 1e-8 tolerance 
    # but Newton can polish it faster if close.
    if best_val > 1e-8 and _time_left() > 1.0 and counter[0] < _CALL_BUDGET - 10:
        x = best_x
        for _ in range(5):
            if _time_left() < 0.5 or counter[0] >= _CALL_BUDGET:
                break
            fx, stop = _counted_call(f, x, counter)
            if stop or not math.isfinite(fx):
                break
            if abs(fx) <= 1e-8:
                return x
            # Finite difference derivative
            h = 1e-8 * max(1.0, abs(x))
            fx_ph, stop = _counted_call(f, x + h, counter)
            if stop or not math.isfinite(fx_ph):
                break
            d = (fx_ph - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            # Check if x_new is better
            fx_new, stop = _counted_call(f, x_new, counter)
            if stop or not math.isfinite(fx_new):
                break
            if abs(fx_new) < abs(fx):
                x = x_new
            else:
                break
        # Return best x found in this loop
        if _time_left() > 0.5:
            fx_final, _ = _counted_call(f, x, counter)
            if math.isfinite(fx_final) and abs(fx_final) < best_val:
                return x

    return best_x
