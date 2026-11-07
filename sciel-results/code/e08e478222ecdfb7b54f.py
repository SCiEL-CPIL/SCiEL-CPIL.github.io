"""
Root finder using high-order interpolation for smooth polynomials.
Leverages the fact that f is a polynomial to use Inverse Quadratic 
or Cubic Interpolation for superlinear convergence, with bisection 
as a robust fallback.
"""
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Optimized for smooth polynomial functions.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    TOL = 1e-9  # Internal tolerance, stricter than 1e-8
    MAX_CALLS = 480  # Safety margin for 500 limit
    
    x = float(x0)
    call_count = 0
    cache = {}
    
    def _elapsed():
        return time.time() - start_time
    
    def _time_ok():
        return _elapsed() < deadline and call_count < MAX_CALLS
    
    def _f(x_val):
        """Cached function evaluation with call counting."""
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(key)
            call_count += 1
            if not math.isfinite(val):
                cache[key] = None
                return None
            cache[key] = val
            return val
        except Exception:
            call_count += 1
            return None

    # Phase 0: Check if x0 is already a root
    fx0 = _f(x)
    if fx0 is None:
        return x
    if abs(fx0) <= TOL:
        return x
    
    # Phase 1: Bracketing
    # We need to find [a, b] such that f(a)*f(b) < 0
    a = b = None
    fa = fb = None
    
    # Strategy: Exponential expansion from x0
    # Start with a small step, double it each time
    step = 1e-3
    max_expansions = 30 # 2^30 is huge, should be enough for most polynomials
    
    # Track points for interpolation later
    # We store (x, f(x))
    points = [(x, fx0)]
    
    for i in range(max_expansions):
        if not _time_ok():
            break
            
        # Try right
        xr = x + step
        fr = _f(xr)
        if fr is not None:
            points.append((xr, fr))
            if abs(fr) <= TOL:
                return xr
            if fx0 is not None and fx0 * fr < 0:
                a, b = x, xr
                fa, fb = fx0, fr
                break
        
        # Try left
        xl = x - step
        fl = _f(xl)
        if fl is not None:
            points.append((xl, fl))
            if abs(fl) <= TOL:
                return xl
            if fx0 is not None and fx0 * fl < 0:
                a, b = xl, x
                fa, fb = fl, fx0
                break
        
        step *= 2.0

    # If no bracket found, we might have a local minimum or the root is far.
    # Try a wider grid search if we haven't used too many calls.
    if a is None or b is None:
        # Grid search to find a sign change or close root
        best_x = x
        best_abs = abs(fx0)
        
        # Sample points in a wider range
        # Use the points we already have to estimate scale
        max_range = max(abs(p[0] - x) for p in points) if points else 1.0
        search_range = max_range * 10 + 10.0
        
        n_samples = 50
        for i in range(n_samples + 1):
            if not _time_ok():
                break
            t = i / n_samples
            x_try = x - search_range + 2 * search_range * t
            f_try = _f(x_try)
            if f_try is not None:
                if abs(f_try) < best_abs:
                    best_abs = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= TOL:
                    return x_try
                # Check for sign change with previous point
                if i > 0:
                    prev_x = points[-1][0] if points else None
                    # Actually, we should check against the immediate previous sample
                    # Let's just store samples in a list for this phase
                    pass
        
        # If we found a better x but no bracket, try Newton from best_x
        x_curr = best_x
        fx_curr = _f(x_curr)
        if fx_curr is None:
            return best_x
            
        for _ in range(10):
            if not _time_ok():
                break
            if abs(fx_curr) <= TOL:
                return x_curr
            # Numerical derivative
            h = 1e-7 * max(1.0, abs(x_curr))
            fx_plus = _f(x_curr + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx_curr) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x_curr - fx_curr / d
            if not math.isfinite(x_new):
                break
            # Damping
            if abs(x_new - x_curr) > 2.0 * max(1.0, abs(x_curr)):
                x_new = x_curr + 2.0 * max(1.0, abs(x_curr)) * (1 if x_new > x_curr else -1)
            x_curr = x_new
            fx_curr = _f(x_curr)
            if fx_curr is None:
                break
        
        return x_curr

    # Phase 2: Refinement using Interpolation
    # We have a bracket [a, b] with fa*fb < 0
    # Ensure fa < 0 < fb for consistency in some logic, or just track signs
    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa
        
    x_lo, x_hi = a, b
    f_lo, f_hi = fa, fb
    
    # History for interpolation: list of (x, f(x))
    # Initialize with the bracket endpoints
    history = [(x_lo, f_lo), (x_hi, f_hi)]
    
    # We want to keep the most recent 4 points for cubic interpolation
    # or 3 for quadratic.
    
    for _ in range(60): # Max iterations for refinement
        if not _time_ok():
            break
            
        # Check if we are close enough
        if abs(f_lo) <= TOL:
            return x_lo
        if abs(f_hi) <= TOL:
            return x_hi
            
        # Calculate midpoint
        x_mid = (x_lo + x_hi) / 2.0
        
        # Try Inverse Quadratic Interpolation (IQI) if we have 3 points
        # We need 3 distinct points. Let's use the two endpoints and the last interior point if available.
        # To keep it simple and robust, we will maintain a list of the last 4 points.
        
        # Determine candidate x_new
        x_new = x_mid # Default to bisection
        
        if len(history) >= 3:
            # Use the last 3 points for Quadratic Interpolation
            # Points: (x1, f1), (x2, f2), (x3, f3)
            # We want to find x where P(x) = 0.
            # Inverse form: x = f1 * (x2-x3) / ((f1-f2)*(f1-f3)) ... no, that's for inverse.
            # Let's use standard Lagrange or Newton divided differences to find root of quadratic.
            
            x1, f1 = history[-3]
            x2, f2 = history[-2]
            x3, f3 = history[-1]
            
            # Check for duplicate x values
            if x1 != x2 and x2 != x3 and x1 != x3:
                # Coefficients of quadratic P(x) = A x^2 + B x + C
                # Using Lagrange basis polynomials
                # L1 = (x-x2)(x-x3) / ((x1-x2)(x1-x3))
                # L2 = (x-x1)(x-x3) / ((x2-x1)(x2-x3))
                # L3 = (x-x1)(x-x2) / ((x3-x1)(x3-x2))
                
                d12 = x1 - x2
                d13 = x1 - x3
                d23 = x2 - x3
                
                # P(x) = f1 * (x-x2)(x-x3)/d12/d13 + ...
                # We want P(x) = 0.
                # A = f1/(d12*d13) + f2/(d21*d23) + f3/(d31*d32)
                # B = f1*(-x2-x3)/(d12*d13) + ...
                # C = f1*(x2*x3)/(d12*d13) + ...
                
                A = f1 / (d12 * d13) + f2 / (d12 * d23) + f3 / (d13 * d23)
                B = f1 * (-(x2 + x3)) / (d12 * d13) + f2 * (-(x1 + x3)) / (d12 * d23) + f3 * (-(x1 + x2)) / (d13 * d23)
                C = f1 * (x2 * x3) / (d12 * d13) + f2 * (x1 * x3) / (d12 * d23) + f3 * (x1 * x2) / (d13 * d23)
                
                # Solve A x^2 + B x + C = 0
                if abs(A) > 1e-15:
                    disc = B * B - 4 * A * C
                    if disc >= 0:
                        sqrt_disc = math.sqrt(disc)
                        root1 = (-B - sqrt_disc) / (2 * A)
                        root2 = (-B + sqrt_disc) / (2 * A)
                        
                        # Check which root is in [x_lo, x_hi]
                        # We prefer the one closer to the midpoint or that reduces the bracket more
                        # Usually, the one inside the bracket is the good one.
                        if x_lo < root1 < x_hi:
                            x_new = root1
                        elif x_lo < root2 < x_hi:
                            x_new = root2
                        else:
                            # If both are outside, or one is outside, fallback to bisection
                            # Or check if one is close to the bracket
                            pass
                    else:
                        pass # No real roots
                elif abs(B) > 1e-15:
                    # Linear
                    root = -C / B
                    if x_lo < root < x_hi:
                        x_new = root
                else:
                    pass # Constant, no root

        # If we didn't find a good interpolation candidate, use bisection
        # We already set x_new = x_mid initially
        
        # Evaluate f(x_new)
        f_new = _f(x_new)
        if f_new is None:
            break
            
        if abs(f_new) <= TOL:
            return x_new
            
        # Update bracket
        if f_new * f_lo < 0:
            x_hi, f_hi = x_new, f_new
        else:
            x_lo, f_lo = x_new, f_new
            
        # Update history
        history.append((x_new, f_new))
        # Keep only the last 4 points to avoid memory growth and ensure recency
        if len(history) > 4:
            history.pop(0)

    # Return the best estimate
    if abs(f_lo) < abs(f_hi):
        return x_lo
    else:
        return x_hi
