"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Robust derivative-free root finder using Inverse Quadratic Interpolation (IQI)
    with bracketing fallback. Optimized for smooth polynomials.
    """
    x = float(x0)
    fx = f(x)
    
    # Check if x0 is already a root
    if abs(fx) <= 1e-8:
        return x

    # Step 1: Find a bracket [a, b] such that f(a) * f(b) <= 0
    # Start with a small step and expand
    h = 1e-4 * max(1.0, abs(x))
    a = x
    fa = fx
    
    # Try to find a point b with opposite sign
    b = None
    fb = None
    # Try positive direction
    for i in range(50):
        x_try = a + h
        f_try = f(x_try)
        if abs(f_try) <= 1e-8:
            return x_try
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
        h *= 2.0
        if not math.isfinite(x_try) or not math.isfinite(f_try):
            break
    
    if b is None:
        # Try negative direction
        h = 1e-4 * max(1.0, abs(x))
        for i in range(50):
            x_try = a - h
            f_try = f(x_try)
            if abs(f_try) <= 1e-8:
                return x_try
            if fa * f_try < 0:
                b = x_try
                fb = f_try
                break
            h *= 2.0
            if not math.isfinite(x_try) or not math.isfinite(f_try):
                break
    
    if b is None:
        # No bracket found, fall back to secant or just return best guess?
        # For polynomials, a bracket should exist unless the root is complex or we are far away.
        # Let's try a secant step from a and a+h (last tried)
        x_try = a + h
        f_try = f(x_try)
        if abs(f_try) <= 1e-8:
            return x_try
        # Secant step
        if fa != f_try:
            x_sec = a - fa * (x_try - a) / (f_try - fa)
            f_sec = f(x_sec)
            if abs(f_sec) <= 1e-8:
                return x_sec
            # Continue with a simple iteration? Or return best?
            # Given the constraints, if we can't bracket, we might fail.
            # But for "smooth polynomial equations", a bracket is almost always findable.
            # Let's just return the point with smallest |f| found so far as a last resort.
            if abs(f_try) < abs(fa):
                return x_try
            return a

    # We have a bracket [a, b]
    # We need a third point for IQI. Let's use the midpoint.
    c = (a + b) / 2.0
    fc = f(c)
    if abs(fc) <= 1e-8:
        return c
    
    # Maintain a history of points for IQI selection
    # We will use the bracket endpoints and the most recent point for the IQI fit.
    # This ensures the fit is well-conditioned and uses global info.
    
    max_iter = 50
    tol = 1e-8
    
    # Current bracket endpoints
    # a, fa and b, fb are the bracket.
    # last_x, last_f is the most recent point (initially c, fc)
    last_x = c
    last_f = fc
    
    for _ in range(max_iter):
        # Check if any point is a root
        if abs(fa) <= tol: return a
        if abs(fb) <= tol: return b
        if abs(last_f) <= tol: return last_x

        # Select 3 points for IQI:
        # 1. a, fa
        # 2. b, fb
        # 3. last_x, last_f
        
        x1, f1 = a, fa
        x2, f2 = b, fb
        x3, f3 = last_x, last_f
        
        # Ensure distinct x
        if abs(x1 - x2) < 1e-15 or abs(x2 - x3) < 1e-15 or abs(x1 - x3) < 1e-15:
            # If points are too close, use bisection
            x_new = (a + b) / 2.0
            f_new = f(x_new)
            if abs(f_new) <= tol:
                return x_new
            # Update bracket
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new
            last_x = x_new
            last_f = f_new
            continue
        
        # Compute IQI step
        # Fit quadratic q(f) = A*f^2 + B*f + C through (f1, x1), (f2, x2), (f3, x3)
        # Solve for x when f=0 => x = C
        
        d1 = f1 - f3
        d2 = f2 - f3
        
        if abs(d1) < 1e-15 or abs(d2) < 1e-15:
            # Degenerate, use Bisection
            x_new = (a + b) / 2.0
        else:
            M1 = d1 * (f1 + f3)
            M2 = d2 * (f2 + f3)
            R1 = x1 - x3
            R2 = x2 - x3
            
            det = M1 * d2 - M2 * d1
            
            if abs(det) < 1e-15:
                # Degenerate, use Bisection
                x_new = (a + b) / 2.0
            else:
                A = (R1 * d2 - R2 * d1) / det
                B = (M1 * R2 - M2 * R1) / det
                C = x3 - A * f3 * f3 - B * f3
                x_new = C
        
        # Check if x_new is within the current bracket [a, b]
        if not (min(a, b) <= x_new <= max(a, b)):
            # If outside, use Bisection to be safe
            x_new = (a + b) / 2.0
            
        # Evaluate new point
        f_new = f(x_new)
        if abs(f_new) <= tol:
            return x_new
        
        # Update Bracket
        if fa * f_new < 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new
            
        # Update last point
        last_x = x_new
        last_f = f_new
        
    # Final check
    if abs(fa) <= tol: return a
    if abs(fb) <= tol: return b
    if abs(last_f) <= tol: return last_x
    
    # Return the point with smallest |f(x)|
    candidates = [(a, fa), (b, fb), (last_x, last_f)]
    return min(candidates, key=lambda p: abs(p[1]))[0]
# EVOLVE-BLOCK-END
