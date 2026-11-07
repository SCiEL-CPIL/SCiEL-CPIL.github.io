"""
Strategy: Global IQI Scan with Local Refinement.
1. Perform a broad linear scan around x0 to identify the most promising root region.
   - Estimate a reasonable step size based on a quick frequency probe.
   - Scan a fixed range (e.g., ±50) with limited calls (e.g., 150).
   - Track the best bracket (sign change) or closest point to zero.
2. Use the best identified bracket or neighborhood to run a high-precision 
   Inverse Quadratic Interpolation (IQI) solver.
   - IQI achieves superlinear convergence with 1 function eval per step.
   - Falls back to Brent's method if IQI stalls or fails.
3. This approach ensures we don't waste calls on exponential expansion if the root
   is nearby, and it handles oscillatory functions better by using an informed step size.
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

def _iqi_step(x0, f0, x1, f1, x2, f2):
    """
    Perform one Inverse Quadratic Interpolation step.
    Returns the next x estimate.
    """
    # If any f values are identical, fall back to secant or bisection logic
    if f0 == f1 or f1 == f2 or f0 == f2:
        # Fall back to secant on the pair with smallest |f|
        if abs(f0) < abs(f2):
            # Secant on (x0, x1)
            if f1 == f0: return x0
            return x0 - f0 * (x1 - x0) / (f1 - f0)
        else:
            # Secant on (x1, x2)
            if f2 == f1: return x1
            return x1 - f1 * (x2 - x1) / (f2 - f1)
    
    # Standard IQI formula
    # We want to find x such that f(x) = 0.
    # We model f(x) as a quadratic passing through (x0,f0), (x1,f1), (x2,f2).
    # Actually, it's easier to model x as a function of f (inverse quadratic).
    # x = A*f^2 + B*f + C
    
    # Lagrange interpolation for x(f)
    # x(f) = f0^2 * (f - f1)(f - f2) / ((f0-f1)(f0-f2)) * x0 + ...
    
    d1 = (f0 - f1) * (f0 - f2)
    d2 = (f1 - f0) * (f1 - f2)
    d3 = (f2 - f0) * (f2 - f1)
    
    if d1 == 0 or d2 == 0 or d3 == 0:
        return None
        
    # We want x_next where f(x_next) = 0
    # So we evaluate the interpolating polynomial at f=0
    x_next = (f0**2 * (0 - f1) * (0 - f2) / d1) * x0 + \
             (f1**2 * (0 - f0) * (0 - f2) / d2) * x1 + \
             (f2**2 * (0 - f0) * (0 - f1) / d3) * x2
             
    return x_next

def _iqi_solve(f, x0, x1, x2, f0, f1, f2, counter, max_iter=20):
    """
    Solve using IQI, maintaining a bracket if possible.
    x0, x1, x2 are the last 3 points.
    We need to maintain a bracket [a, b] where f(a)*f(b) < 0.
    """
    # Identify the current bracket
    # We assume the caller passes points that form or are near a bracket.
    # For robustness, let's re-establish a bracket from the 3 points if possible.
    
    points = [(x0, f0), (x1, f1), (x2, f2)]
    
    # Find a bracket among these 3
    bracket = None
    for i in range(3):
        for j in range(i+1, 3):
            xi, fi = points[i]
            xj, fj = points[j]
            if math.isfinite(fi) and math.isfinite(fj) and fi * fj < 0:
                bracket = (xi, xj)
                break
        if bracket:
            break
            
    if not bracket:
        # If no bracket, pick the point with smallest |f| and try to expand locally
        # This is a fallback, IQI prefers brackets
        best_idx = min(range(3), key=lambda i: abs(points[i][1]) if math.isfinite(points[i][1]) else float('inf'))
        best_x, best_f = points[best_idx]
        if abs(best_f) <= 1e-8:
            return best_x
        # Try a small step
        step = 0.01
        x_new = best_x + step
        f_new = _counted_call(f, x_new, counter)
        if math.isfinite(f_new) and best_f * f_new < 0:
            bracket = (best_x, x_new)
        else:
            x_new = best_x - step
            f_new = _counted_call(f, x_new, counter)
            if math.isfinite(f_new) and best_f * f_new < 0:
                bracket = (x_new, best_x)
            else:
                # No bracket found, return best guess
                return best_x
                
    a, b = bracket
    
    # Ensure we have 3 distinct points for IQI, or initialize them
    # We need x0, x1, x2 with f0, f1, f2
    # Let's collect the unique points we have
    used_points = set()
    hist_x = []
    hist_f = []
    
    for x, fx in [(x0, f0), (x1, f1), (x2, f2), (a, f(a) if a not in [x0,x1,x2] else f0 if a==x0 else f1 if a==x1 else f2), (b, f(b) if b not in [x0,x1,x2] else f0 if b==x0 else f1 if b==x1 else f2)]:
         pass # This logic is getting messy. Let's simplify.
         
    # Simpler approach: Just use the 3 points provided, and if IQI steps out of bracket, use bisection.
    # We need to maintain a list of last 3 points.
    
    xs = [x0, x1, x2]
    fs = [f0, f1, f2]
    
    # Ensure bracket is respected
    # If current best x is not in bracket, we might need to restart with bracket points.
    # Let's just run IQI on the 3 points, but if the new x is outside [a,b], use bisection.
    
    for _ in range(max_iter):
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
            break
            
        x_new = _iqi_step(xs[0], fs[0], xs[1], fs[1], xs[2], fs[2])
        
        if x_new is None or not math.isfinite(x_new):
            # Fallback to bisection
            mid = (a + b) / 2.0
            x_new = mid
            
        # Check if x_new is in bracket
        if not (a <= x_new <= b):
            x_new = (a + b) / 2.0 # Bisection fallback
            
        f_new = _counted_call(f, x_new, counter)
        
        if not math.isfinite(f_new):
            # If non-finite, shrink bracket? Or just break?
            # For oscillatory, non-finite is rare unless singular.
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update bracket
        if fs[1] * f_new < 0:
            a, b = xs[1], x_new
        elif fs[2] * f_new < 0:
            a, b = xs[2], x_new
        # Note: IQI can be tricky with brackets. 
        # A safer hybrid is Brent's method which handles this automatically.
        # Given the complexity of implementing robust IQI with bracket maintenance,
        # and that Brent's is already very efficient (superlinear), 
        # the main gain of this strategy is the GLOBAL SCAN part.
        # Let's just use Brent's for the local refinement, as it's robust and fast enough.
        
        # Shift history
        xs.pop(0)
        fs.pop(0)
        xs.append(x_new)
        fs.append(f_new)
        
    # If loop finishes, return best point
    best_x = min(xs, key=lambda x: abs(f(x)) if math.isfinite(f(x)) else float('inf'))
    return best_x

def _global_scan(f, x0, counter):
    """
    Scan a range around x0 to find the best bracket.
    Returns (best_a, best_b) or None.
    """
    # Estimate step size
    # Probe a few points to guess frequency
    probe_step = 0.5
    f0 = _counted_call(f, x0, counter)
    f1 = _counted_call(f, x0 + probe_step, counter)
    
    # If sign change in probe, step is too big or root is very close.
    # Let's assume a standard oscillation period of ~1-10 units.
    # A step of 0.1 is safe for high freq, 1.0 for low freq.
    # Let's use 0.1 as a conservative scan step for the first 10 units,
    # then expand.
    
    best_bracket = None
    best_min_abs_f = float('inf')
    
    # Phase 1: Dense scan near x0 (±5)
    # 100 points in 10 units -> step 0.1
    # Calls: 100
    # We already used 2 calls. Budget left ~480.
    
    # Let's do a uniform scan from x0-10 to x0+10 with step 0.2 (100 points)
    # If no sign change, expand to ±50 with step 1.0 (100 points)
    
    def scan_range(a, b, n_points):
        nonlocal best_bracket, best_min_abs_f
        xs = np.linspace(a, b, n_points)
        prev_x = None
        prev_f = None
        
        for x in xs:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 50:
                return
                
            fx = _counted_call(f, x, counter)
            
            if not math.isfinite(fx):
                prev_x = None
                prev_f = None
                continue
                
            if abs(fx) < best_min_abs_f:
                best_min_abs_f = abs(fx)
                
            if prev_x is not None and math.isfinite(prev_f):
                if prev_f * fx < 0:
                    # Found a bracket
                    # Check if it's "better" (tighter or smaller |f|)
                    # For now, just take the first one found in the dense scan
                    # But we want the one closest to x0? Or the one with smallest |f|?
                    # Usually, the first one found in a scan starting from x0 is good.
                    # Let's store it.
                    if best_bracket is None:
                        best_bracket = (prev_x, x)
                        # We could stop here, but let's scan a bit more to see if there's a closer one?
                        # No, scanning is expensive. Let's stop at the first sign change 
                        # in the dense region.
                        return
            
            prev_x = x
            prev_f = fx

    # Scan ±10 with step 0.1 (200 points)
    # This is 200 calls.
    # If found, we are done with search.
    scan_range(x0 - 10, x0 + 10, 200)
    
    if best_bracket is not None:
        return best_bracket
        
    # If not found, scan ±100 with step 1.0 (200 points)
    # Total calls so far ~202.
    # Remaining ~280.
    scan_range(x0 - 100, x0 + 100, 200)
    
    return best_bracket

def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy:
    1. Global scan to find a bracket.
    2. Brent's method to converge.
    """
    counter = [0]
    x0 = float(x0)
    
    # Check x0 first
    f0 = _counted_call(f, x0, counter)
    if math.isfinite(f0) and abs(f0) <= 1e-8:
        return x0
        
    # Global Scan
    bracket = _global_scan(f, x0, counter)
    
    if bracket is not None:
        a, b = bracket
        # Use Brent's method for robust convergence
        try:
            result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(f(result)) <= 1e-8:
                return result
        except Exception:
            pass
            
    # Fallback: If no bracket found, return x with smallest |f| seen?
    # We didn't track the best x in _global_scan if no bracket was found.
    # Let's do a final desperate local search around x0
    # Or just return x0.
    
    # Actually, let's refine _global_scan to return the best x even if no bracket.
    # But for now, if no bracket, we fail.
    # To improve, let's modify _global_scan to track best_x.
    
    # Re-implementing logic to ensure we return something valid if possible.
    # If _global_scan returns None, it means no sign change was found.
    # This is a failure case for this strategy.
    # However, we can try a few random points or a local quadratic fit.
    
    # Let's just return x0 as a last resort, though it will likely fail the |f(x)| check.
    return x0

# Refinement: The _global_scan above discards the best_x if no bracket is found.
# Let's fix that.

def _global_scan_v2(f, x0, counter):
    """
    Scan a range around x0 to find the best bracket or best point.
    Returns (type, value1, value2) where type is 'bracket' or 'point'.
    """
    best_bracket = None
    best_point = None
    best_min_abs_f = float('inf')
    
    def scan_range(a, b, n_points):
        nonlocal best_bracket, best_point, best_min_abs_f
        xs = np.linspace(a, b, n_points)
        prev_x = None
        prev_f = None
        
        for x in xs:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 50:
                return
                
            fx = _counted_call(f, x, counter)
            
            if not math.isfinite(fx):
                prev_x = None
                prev_f = None
                continue
                
            if abs(fx) < best_min_abs_f:
                best_min_abs_f = abs(fx)
                best_point = x
                
            if prev_x is not None and math.isfinite(prev_f):
                if prev_f * fx < 0:
                    # Found a bracket
                    if best_bracket is None:
                        best_bracket = (prev_x, x)
                        return # Stop scanning this range
            
            prev_x = x
            prev_f = fx

    # Scan ±10 with step 0.1 (200 points)
    scan_range(x0 - 10, x0 + 10, 200)
    
    if best_bracket is not None:
        return ('bracket', best_bracket[0], best_bracket[1])
        
    # If not found, scan ±100 with step 1.0 (200 points)
    scan_range(x0 - 100, x0 + 100, 200)
    
    if best_bracket is not None:
        return ('bracket', best_bracket[0], best_bracket[1])
        
    if best_point is not None:
        return ('point', best_point, None)
        
    return None

def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy:
    1. Global scan to find a bracket or best point.
    2. Brent's method to converge if bracket found.
    3. If only point found, try to create a bracket around it.
    """
    counter = [0]
    x0 = float(x0)
    
    # Check x0 first
    f0 = _counted_call(f, x0, counter)
    if math.isfinite(f0) and abs(f0) <= 1e-8:
        return x0
        
    # Global Scan
    result = _global_scan_v2(f, x0, counter)
    
    if result:
        rtype, v1, v2 = result
        
        if rtype == 'bracket':
            a, b = v1, v2
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if abs(f(root)) <= 1e-8:
                    return root
            except Exception:
                pass
                
        elif rtype == 'point':
            x_best = v1
            f_best = _counted_call(f, x_best, counter)
            if abs(f_best) <= 1e-8:
                return x_best
                
            # Try to bracket around x_best
            # Expand step from 0.1 to 10
            step = 0.1
            for _ in range(10):
                if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
                    break
                    
                a = x_best - step
                b = x_best + step
                fa = _counted_call(f, a, counter)
                fb = _counted_call(f, b, counter)
                
                if math.isfinite(fa) and math.isfinite(fb):
                    if fa * fb < 0:
                        try:
                            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                            if abs(f(root)) <= 1e-8:
                                return root
                        except Exception:
                            pass
                    if abs(fa) <= 1e-8: return a
                    if abs(fb) <= 1e-8: return b
                    
                step *= 2.0
            
            return x_best
            
    return x0
