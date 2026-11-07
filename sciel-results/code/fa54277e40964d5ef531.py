import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 3 equations + overhead
    
    call_count = 0
    max_calls = 500
    best_x = x0
    best_f = float('inf')
    
    def f_counted(x):
        nonlocal call_count, best_x, best_f
        if call_count >= max_calls:
            raise Exception("Max calls exceeded")
        call_count += 1
        val = f(x)
        if abs(val) < abs(best_f):
            best_f = val
            best_x = x
        return val
    
    # Check if x0 is already a root
    fx0 = f_counted(x0)
    if abs(fx0) <= 1e-8:
        return x0
    
    # Strategy 1: Try cubic reconstruction with 4 points
    # We need 4 distinct points. Let's pick points around x0.
    # For a polynomial of degree <= 3, 4 points suffice.
    
    points = [x0, x0 + 1.0, x0 + 2.0, x0 + 3.0]
    # Ensure distinctness
    if points[0] == points[1]:
        points = [x0, x0 + 0.1, x0 + 0.2, x0 + 0.3]
    
    try:
        if time.time() > deadline:
            return best_x
        vals = [f_counted(p) for p in points]
        
        # Check if any point is a root
        for p, v in zip(points, vals):
            if abs(v) <= 1e-8:
                return p
        
        # Interpolate cubic polynomial
        # Using numpy.polyfit
        coeffs = np.polyfit(points, vals, 3)
        # Find roots of the cubic
        roots = np.roots(coeffs)
        
        # Filter for real roots
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        
        if not real_roots:
            raise ValueError("No real roots from cubic")
        
        # Find the root closest to x0
        best_root = min(real_roots, key=lambda r: abs(r - x0))
        
        # Verify the root
        f_best = f_counted(best_root)
        if abs(f_best) <= 1e-8:
            return best_root
        else:
            # The cubic approximation was not accurate enough or degree > 3
            # We have used 5 calls so far (1 for x0 + 4 for points + 1 for verify = 6 calls)
            # We can proceed to other methods.
            pass
            
    except Exception:
        pass
    
    # Strategy 2: Bracketing with Brent's method
    # We need to find a bracket [a, b] such that f(a)*f(b) < 0.
    # We will expand the interval from x0.
    
    # We already have fx0.
    # Let's try expanding in both directions.
    # We'll use a geometric expansion.
    
    a = x0
    b = x0
    fa = fx0
    fb = fx0
    
    # Try small steps first
    step = 1e-2
    found_bracket = False
    last_a, last_b, last_fa, last_fb = None, None, None, None
    
    # We have used some calls. Let's say we have budget left.
    # We'll iterate until we find a bracket or exceed time/calls.
    
    # To save calls, we can reuse values if possible, but since points are different, we can't.
    
    for i in range(50):
        if time.time() > deadline or call_count > max_calls - 10:
            break
            
        # Try right side
        b = x0 + step
        fb = f_counted(b)
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            found_bracket = True
            last_a, last_b, last_fa, last_fb = x0, b, fa, fb
            break
            
        # Try left side
        a = x0 - step
        fa = f_counted(a)
        if abs(fa) <= 1e-8:
            return a
        if fa * fx0 < 0:
            found_bracket = True
            last_a, last_b, last_fa, last_fb = a, x0, fa, fx0
            break
            
        step *= 2.0
        
    if found_bracket:
        try:
            root = brentq(f_counted, last_a, last_b, xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except Exception:
            pass
    
    # Strategy 3: Newton's Method / Secant Method
    # If no bracket found, try Newton.
    # We need a derivative. We can approximate it or use secant.
    # Secant uses 2 previous points. We have x0 and fx0.
    # Let's pick another point.
    
    x1 = x0 + 1e-4
    if call_count < max_calls - 1:
        fx1 = f_counted(x1)
    else:
        fx1 = fx0 # Fallback
    
    # Secant method
    x_prev = x0
    x_curr = x1
    f_prev = fx0
    f_curr = fx1
    
    for _ in range(50):
        if time.time() > deadline or call_count > max_calls - 1:
            break
            
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            break
            
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        # Check for convergence or divergence
        if not math.isfinite(x_new):
            break
            
        # Damping
        if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
             x_new = x_curr + 10 * max(1.0, abs(x_curr)) * (1 if x_new > x_curr else -1)
             
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        
        if call_count < max_calls:
            f_curr = f_counted(x_curr)
        else:
            break
            
        if abs(f_curr) <= 1e-8:
            return x_curr
            
    # Return the best estimate
    if abs(f_curr) < abs(f_prev):
        return x_curr
    else:
        return x_prev
