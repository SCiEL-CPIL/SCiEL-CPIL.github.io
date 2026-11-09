"""Root finder for smooth polynomial equations with call-count optimization."""
import time

# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for a smooth polynomial.
    
    Uses Illinois method (modified bisection with secant step) for robustness and speed.
    """
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    tol = 1e-13  # Tighter tolerance to ensure 1e-8 pass
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x
        
    if not math.isfinite(fx):
        # Try nearby points
        for dx in [1e-4, -1e-4, 1e-2, -1e-2, 1.0, -1.0]:
            try:
                x_try = x + dx
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except Exception:
                continue
        else:
            return x

    if abs(fx) <= tol:
        return x

    # 2. Find a bracket [a, b] such that f(a) * f(b) < 0
    # Start with a small step
    step = 1.0
    a = x
    fa = fx
    
    # Try positive direction first
    b = x + step
    try:
        fb = f(b)
    except Exception:
        fb = float('inf')
        
    # If no sign change, try negative direction
    if not (math.isfinite(fb) and fa * fb < 0):
        b = x - step
        try:
            fb = f(b)
        except Exception:
            fb = float('inf')
            
    # Expand bracket if necessary
    if not (math.isfinite(fb) and fa * fb < 0):
        # Geometric expansion
        for i in range(30):
            if time.time() > deadline:
                break
            step *= 2.0
            # Try expanding from x
            b_pos = x + step
            try:
                fb_pos = f(b_pos)
                if math.isfinite(fb_pos) and fa * fb_pos < 0:
                    b = b_pos
                    fb = fb_pos
                    break
            except Exception:
                pass
            
            b_neg = x - step
            try:
                fb_neg = f(b_neg)
                if math.isfinite(fb_neg) and fa * fb_neg < 0:
                    b = b_neg
                    fb = fb_neg
                    break
            except Exception:
                pass
        
        if not (math.isfinite(fb) and fa * fb < 0):
            # If we can't bracket, return the point with smallest |f(x)| found so far
            # We only have x and maybe a few others. Just return x.
            return x

    # 3. Illinois Method
    # Invariant: f(a) * f(b) < 0
    # We maintain |f(a)| >= |f(b)| to bias the search
    
    for i in range(100):
        if time.time() > deadline:
            break
            
        # Ensure |fa| >= |fb|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            
        # Secant step
        if fa != fb:
            x_new = b - fb * (b - a) / (fb - fa)
        else:
            x_new = (a + b) / 2.0
            
        # If x_new is outside (a, b), use bisection
        if x_new <= a or x_new >= b:
            x_new = (a + b) / 2.0
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(fx_new):
            # If non-finite, shrink bracket? Or just break.
            # For polynomials, this shouldn't happen unless we go to infinity.
            break
            
        if abs(fx_new) <= tol:
            return x_new
            
        # Update bracket
        if fa * fx_new <= 0:
            b = x_new
            fb = fx_new
        else:
            a = x_new
            fa = fx_new
            
        # Check for convergence of interval
        if abs(b - a) < 1e-15:
            break
            
    return x_new if 'x_new' in locals() and math.isfinite(x_new) else x

# EVOLVE-BLOCK-END
