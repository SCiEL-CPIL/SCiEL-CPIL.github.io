"""
Strategy: Exponential Bracketing Expansion with Brent's Method.
1. Start at x0.
2. Expand the search interval by doubling its size (right, then left) until a sign change is detected.
   This is O(log(distance_to_root)) calls, which is much more efficient than linear scans for distant roots.
3. Once a bracket [a, b] with f(a)*f(b) < 0 is found, use scipy.optimize.brentq to converge.
   Brent's method is robust for oscillatory functions and converges quickly (cubic convergence).
4. If no bracket is found within budget, fall back to a dense scan with adaptive step size.
This approach minimizes wasted calls on empty regions and focuses computational effort on the convergence phase.
"""
import math
import time
import numpy as np
from scipy.optimize import brentq

# Global deadline: evaluator kills after 150s wall-clock.
_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 490  # stay under 500


def _time_left():
    return _DEADLINE - time.time()


def _counted_call(f, x, counter):
    """Call f(x), track count, guard against exceptions / non-finite."""
    counter[0] += 1
    try:
        val = f(x)
    except Exception:
        return float('nan')
    if not math.isfinite(val):
        return val
    return val


def _find_bracket_by_expansion(f, x0, counter):
    """
    Expand search interval from x0 to find a sign-changing bracket.
    Returns (a, b) if bracket found, else None.
    """
    a = x0
    fa = _counted_call(f, a, counter)
    
    if math.isfinite(fa) and abs(fa) <= 1e-8:
        return (a, a)  # Root at x0
    
    # Try expanding to the right first
    step = 0.1
    for _ in range(60):  # Max expansion steps
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 20:
            return None
        
        b = x0 + step
        fb = _counted_call(f, b, counter)
        
        if math.isfinite(fa) and math.isfinite(fb):
            if fa * fb < 0:
                return (a, b)
            if abs(fb) <= 1e-8:
                return (b, b)
        
        # Try expanding to the left
        a_left = x0 - step
        fa_left = _counted_call(f, a_left, counter)
        
        if math.isfinite(fa) and math.isfinite(fa_left):
            if fa * fa_left < 0:
                return (a_left, a)
            if abs(fa_left) <= 1e-8:
                return (a_left, a_left)
        
        step *= 2.0
    
    return None


def _dense_scan_fallback(f, x0, counter):
    """
    Fallback: dense scan with adaptive step size.
    """
    # Estimate frequency by probing
    probes = [0.1, 0.5, 1.0, 5.0, 10.0]
    min_period = float('inf')
    prev_f = _counted_call(f, x0, counter)
    
    for s in probes:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 20:
            break
        
        x = x0 + s
        fx = _counted_call(f, x, counter)
        
        if math.isfinite(prev_f) and math.isfinite(fx):
            if prev_f * fx < 0:
                if s < min_period:
                    min_period = s
        
        prev_f = fx
    
    if min_period < float('inf'):
        period = min_period
    else:
        period = 10.0
    
    if period < 0.01:
        period = 0.01
    
    step = period / 10.0
    
    # Scan a reasonable range
    for radius in [5.0 * period, 50.0 * period]:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
        
        a = x0 - radius
        b = x0 + radius
        n_points = int(2 * radius / step) + 1
        
        if n_points > 150:
            step = 2 * radius / 149
            n_points = 150
        
        xs = np.linspace(a, b, n_points)
        
        prev_x = None
        prev_f = None
        
        for x in xs:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                break
            
            fx = _counted_call(f, x, counter)
            
            if math.isfinite(fx):
                if abs(fx) <= 1e-8:
                    return x
                
                if prev_f is not None and math.isfinite(prev_f):
                    if prev_f * fx < 0:
                        try:
                            result = brentq(f, prev_x, x, xtol=1e-12, rtol=1e-12, maxiter=100)
                            if abs(f(result)) <= 1e-8:
                                return result
                        except Exception:
                            pass
                
                prev_x = x
                prev_f = fx
            else:
                prev_x = None
                prev_f = None
    
    return None


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy:
    1. Try exponential bracketing expansion from x0.
    2. If bracket found, use brentq to converge.
    3. If no bracket found, fall back to dense scan.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Exponential bracketing
    bracket = _find_bracket_by_expansion(f, x0, counter)
    
    if bracket is not None:
        a, b = bracket
        if a == b:
            return a
        
        # Use brentq to converge
        try:
            result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(f(result)) <= 1e-8:
                return result
        except Exception:
            pass
    
    # 2. Fallback: dense scan
    result = _dense_scan_fallback(f, x0, counter)
    if result is not None:
        return result
    
    # 3. Final fallback: return x0
    return x0
