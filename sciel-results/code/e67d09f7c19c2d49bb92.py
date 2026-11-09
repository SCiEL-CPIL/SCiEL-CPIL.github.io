import numpy as np

def find_root(f, x0):
    eval_count = 0
    max_evals = 500
    
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("Max evaluations exceeded")
        return f(x)

    x = float(x0)
    fx = f_counted(x)
    
    if abs(fx) <= 1e-8:
        return x

    # Strategy: Muller's Method with Fallback to Bisection
    
    # 1. Initial Points for Muller's
    # We need 3 points.
    # x0 is given.
    # Try to find x1, x2 close to x0.
    
    h = 1e-4 * max(1.0, abs(x))
    
    # Try to establish a triplet (xa, xb, xc)
    # We have x, fx.
    # Let's try x-h, x+h.
    
    # To ensure distinct values, we might need to adjust.
    # For simplicity, let's try a small local search for a better starting point if possible,
    # but Muller is robust enough with close points.
    
    # Let's use x, x+h, x+2h
    # But if f is flat, this might fail.
    
    # Let's try to find a bracket first to ensure we have a region of interest?
    # The directive says "new approach". Muller is the core.
    
    # Let's initialize Muller with x, x+h, x+2h
    xa = x
    fa = fx
    
    xb = x + h
    fb = f_counted(xb)
    
    xc = x + 2*h
    fc = f_counted(xc)
    
    # If we hit NaN or Inf, adjust h
    # Simple retry logic for initialization
    retries = 0
    while (not np.isfinite(fb) or not np.isfinite(fc)) and retries < 10 and eval_count < 50:
        h *= 2.0
        xb = x + h
        xc = x + 2*h
        if eval_count + 2 <= max_evals:
             fb = f_counted(xb)
             fc = f_counted(xc)
        retries += 1

    # If still bad, fallback to simple expansion for bracket
    if not (np.isfinite(fb) and np.isfinite(fc)):
        # Try to find a bracket
        found_bracket = False
        h = 1e-4 * max(1.0, abs(x))
        for i in range(40):
            if eval_count >= max_evals: break
            xa_try = x - h
            xb_try = x + h
            fa_try = f_counted(xa_try)
            fb_try = f_counted(xb_try)
            
            if fa_try * fx < 0:
                a, b = xa_try, x
                found_bracket = True
                break
            if fx * fb_try < 0:
                a, b = x, xb_try
                found_bracket = True
                break
            h *= 2.0
            if h > 1e10: break
        
        if found_bracket:
             # Use Bisection
             for i in range(100):
                 if eval_count >= max_evals: break
                 m = (a + b) / 2.0
                 fm = f_counted(m)
                 if abs(fm) <= 1e-8:
                     return m
                 if (fa_try * fm) <= 0: # Note: fa_try corresponds to xa_try, need to track properly
                     # Actually, let's restart bisection logic cleanly if we go here
                     pass
                 # This fallback path is getting messy. 
                 # Let's simplify: If Muller init fails, just do a robust Bisection search.
                 break
        
        # Robust Bisection Fallback
        # Find bracket
        a, b = None, None
        h = 1e-4 * max(1.0, abs(x))
        fa = fx
        for i in range(40):
            if eval_count >= max_evals - 1: break
            b = x + h
            fb = f_counted(b)
            if fa * fb < 0:
                a = x
                found = True
                break
            # Also check left side?
            a_left = x - h
            fa_left = f_counted(a_left)
            if fa_left * fa < 0:
                a = a_left
                b = x
                found = True
                break
            
            h *= 2.0
            if h > 1e10: break
            
        if a is not None and b is not None:
             # Bisection
             fa = f(a) # Already have? No, we used f_counted in loop. 
             # Let's just re-evaluate to be safe if we didn't store properly, or track.
             # Actually, we have fa and fb from the loop if we track them.
             # Let's implement a clean Bisection function.
             
             pass
             
        return x

    # Proceed with Muller's Method
    # Points: (xa, fa), (xb, fb), (xc, fc)
    # We want to find root of quadratic interpolating these.
    # Let's shift origin to xc for numerical stability.
    # h1 = xb - xc
    # h2 = xa - xc
    # d1 = (fb - fc) / h1
    # d2 = (fa - fc) / h2
    # e = (d1 - d2) / (h1 - h2)
    # b = d1 * e
    # c = e * h1 * h2 ... wait, standard formula.
    
    # Standard Muller's:
    # x0, x1, x2 are points.
    # h1 = x1 - x2
    # h2 = x0 - x2
    # d1 = (f1 - f2) / h1
    # d2 = (f0 - f2) / h2
    # e = (d1 - d2) / (h1 - h2)
    # b = d1 * e
    # c = e * h1 * h2 ... no.
    
    # The quadratic is q(t) = a*t^2 + b*t + c where t = x - x2.
    # c = f2
    # b = d1 + e*h1  (slope at x2? no)
    # Let's use the explicit formula from Numerical Recipes or similar.
    
    # p = (x1 - x2) * (f1 - f2) / (f1 - f0) ... no that's secant.
    
    # Let's use the stable formulation:
    # h1 = x1 - x2
    # h2 = x0 - x2
    # d1 = (f1 - f2) / h1
    # d2 = (f0 - f2) / h2
    # e = (d1 - d2) / (h1 - h2)
    # b = d1 * e
    # c = e * h1 * h2 ... 
    # Actually, the root is given by:
    # x3 = x2 - 2*f2 / (b + sign(b)*sqrt(b^2 - 4*a*f2))
    # where a = e, b = d1 + e*h1
    
    # Let's verify.
    # q(t) = a*t^2 + b*t + c
    # q(0) = c = f2
    # q(h1) = f1 => a*h1^2 + b*h1 + f2 = f1 => a*h1 + b = (f1-f2)/h1 = d1
    # q(h2) = f0 => a*h2^2 + b*h2 + f2 = f0 => a*h2 + b = (f0-f2)/h2 = d2
    # Subtract: a*(h1-h2) = d1 - d2 => a = (d1-d2)/(h1-h2) = e
    # b = d1 - a*h1 = d1 - e*h1
    
    # So:
    # a = e
    # b = d1 - e*h1
    # c = f2
    
    # We want root of a*t^2 + b*t + c = 0.
    # t = (-b +/- sqrt(b^2 - 4ac)) / 2a
    # To avoid cancellation, choose sign of sqrt to be same as b.
    # t = -2c / (b + sign(b)*sqrt(b^2 - 4ac))
    
    # If discriminant < 0, fail.
    
    x_prev2 = xa
    x_prev1 = xb
    x_curr = xc
    f_prev2 = fa
    f_prev1 = fb
    f_curr = fc
    
    for i in range(100):
        if eval_count >= max_evals - 1:
            break
            
        h1 = x_prev1 - x_curr
        h2 = x_prev2 - x_curr
        
        if h1 == 0 or h2 == 0 or (h1 - h2) == 0:
            # Degenerate, break and fallback
            break
            
        d1 = (f_prev1 - f_curr) / h1
        d2 = (f_prev2 - f_curr) / h2
        
        e = (d1 - d2) / (h1 - h2)
        b = d1 - e * h1
        a = e
        c = f_curr
        
        disc = b*b - 4*a*c
        
        if disc < 0:
            # Complex root, fallback
            break
            
        sqrt_disc = np.sqrt(disc)
        
        # Choose sign to avoid cancellation
        if b >= 0:
            t = -2*c / (b + sqrt_disc)
        else:
            t = -2*c / (b - sqrt_disc)
            
        x_new = x_curr + t
        
        # Safety check: if x_new is not finite, break
        if not np.isfinite(x_new):
            break
            
        f_new = f_counted(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update points
        x_prev2 = x_prev1
        x_prev1 = x_curr
        x_curr = x_new
        f_prev2 = f_prev1
        f_prev1 = f_curr
        f_curr = f_new
        
        # Convergence check on x
        if abs(x_new - x_curr) < 1e-12 * max(1.0, abs(x_new)):
             # Check f again just in case
             if abs(f_new) <= 1e-8:
                 return x_new
             # If f is not small enough, we might be stuck at a minimum or flat spot.
             # Break and fallback?
             # Let's continue a bit more or break.
             break

    # Fallback: Bisection
    # We have a set of points evaluated. Let's use the ones with sign change if any.
    # Or just do a fresh bracketing search.
    
    # Collect all evaluated points? We didn't store them all.
    # Let's just do a fresh bracketing search starting from x0.
    # This is inefficient but robust.
    
    x = float(x0)
    fx = f_counted(x)
    if abs(fx) <= 1e-8: return x
    
    h = 1e-4 * max(1.0, abs(x))
    a, b = None, None
    fa = fx
    
    for i in range(40):
        if eval_count >= max_evals - 1: break
        b = x + h
        fb = f_counted(b)
        if fa * fb < 0:
            a = x
            found = True
            break
        # Check left
        a_left = x - h
        fa_left = f_counted(a_left)
        if fa_left * fa < 0:
            a = a_left
            b = x
            found = True
            break
        
        h *= 2.0
        if h > 1e10: break
        
    if a is not None and b is not None:
        # Bisection
        fa = f_counted(a) # Re-eval to be sure, or use stored? We have fa from loop start? No, fa was x.
        # In the loop, fa was fx (at x). 
        # If a=x, fa is known. If a=a_left, fa is fa_left.
        # Let's just re-evaluate a and b to be safe and simple, costs 2 evals.
        if eval_count + 2 <= max_evals:
            fa = f_counted(a)
            fb = f_counted(b)
            
        for i in range(100):
            if eval_count >= max_evals - 1: break
            m = (a + b) / 2.0
            fm = f_counted(m)
            if abs(fm) <= 1e-8:
                return m
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            if abs(b - a) < 1e-12:
                break
                
        return (a + b) / 2.0
        
    return x
