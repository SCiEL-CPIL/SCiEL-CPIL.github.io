"""Root finder for smooth polynomial equations with call-count optimization."""
import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for a smooth polynomial.
    
    Uses a hybrid approach:
    1. Try to bracket the root using expanding intervals.
    2. Use Illinois/bisection for robustness to find a bracket.
    3. Use secant method for fast convergence once bracketed or near root.
    4. Fall back to pure secant/Newton if no bracket found.
    """
    deadline = time.time() + 115.0  # 5s safety margin before 120s limit
    TOL = 1e-8
    BEST_TOL = 1e-12  # aim for tighter to ensure we pass
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    # Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        # Try nearby points
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 0.1, -0.1, 1.0, -1.0]:
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
            return x  # Give up
    
    if not math.isfinite(fx):
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 0.1, -0.1, 1.0, -1.0]:
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
    
    best_x = x
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= TOL:
        return x
    
    # Strategy 1: Try to find a bracket by expanding from x0
    # We look for a sign change
    a = x
    fa = fx
    
    # First, try small steps to find a sign change
    bracketed = False
    b = None
    fb = None
    
    # Try expanding in both directions
    for scale in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline:
            break
            
        # Try positive direction
        x_plus = a + scale * max(1.0, abs(a))
        try:
            f_plus = f(x_plus)
            if math.isfinite(f_plus):
                if fa * f_plus < 0:
                    bracketed = True
                    b = x_plus
                    fb = f_plus
                    break
        except Exception:
            pass
        
        # Try negative direction
        x_minus = a - scale * max(1.0, abs(a))
        try:
            f_minus = f(x_minus)
            if math.isfinite(f_minus):
                if fa * f_minus < 0:
                    bracketed = True
                    b = x_minus
                    fb = f_minus
                    break
        except Exception:
            pass
    
    if bracketed:
        # Use Illinois method (modified bisection) for robustness, then switch to secant
        # Illinois converges linearly but is very robust.
        # We can use it to get close, then use secant for quadratic convergence.
        
        # Ensure a has smaller |f|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        for i in range(100):
            if time.time() > deadline:
                break
                
            if abs(fa) <= TOL:
                best_x = a
                best_abs_fx = abs(fa)
                break
            if abs(fb) <= TOL:
                best_x = b
                best_abs_fx = abs(fb)
                break
                
            # Illinois step
            # If f(a) and f(b) have same sign (shouldn't happen if bracketed correctly), break
            if fa * fb > 0:
                break
                
            c = b - fb * (b - a) / (fb - fa)
            fc = None
            try:
                fc = f(c)
            except Exception:
                # Fall back to bisection
                c = 0.5 * (a + b)
                try:
                    fc = f(c)
                except Exception:
                    break
            
            if not math.isfinite(fc):
                c = 0.5 * (a + b)
                try:
                    fc = f(c)
                except Exception:
                    break
            
            if abs(fc) <= TOL:
                best_x = c
                best_abs_fx = abs(fc)
                break
            
            if abs(fc) < best_abs_fx:
                best_abs_fx = abs(fc)
                best_x = c
                
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = 0.5 * fa  # Illinois modification
                fb = fc
                # Ensure a has smaller |f| for next step? No, Illinois keeps the endpoint with smaller |f| as the "active" one for the secant step, but updates the other.
                # Standard Illinois:
                # c = b - fb*(b-a)/(fb-fa)
                # if f(a)f(c) < 0: b=c, fb=fc
                # else: a=c, fa=0.5*fa, fb=fc  (Wait, usually the one that is NOT updated to the new midpoint gets halved)
                # Let's stick to a simpler robust bisection/secant hybrid:
                # If sign change in [a, c], new interval is [a, c]. Else [c, b].
                # To speed up convergence, use secant step but fall back to bisection if it doesn't reduce interval size enough.
                
        # After Illinois/bisection, if we haven't reached TOL, try secant from the last two points
        if best_abs_fx > TOL and time.time() < deadline:
            # Use the last two points from the loop (a, b)
            # Actually, let's just run a few secant steps from a and b
            x1, f1 = a, fa
            x2, f2 = b, fb
            
            for i in range(20):
                if time.time() > deadline:
                    break
                    
                if abs(f1) <= TOL:
                    best_x = x1
                    best_abs_fx = abs(f1)
                    break
                if abs(f2) <= TOL:
                    best_x = x2
                    best_abs_fx = abs(f2)
                    break
                
                if f1 == f2:
                    break
                    
                x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
                
                if not math.isfinite(x_new):
                    break
                    
                # Dampen if step is too large
                if abs(x_new - x1) > 5.0 * max(1.0, abs(x1)):
                    x_new = x1 + 0.5 * (x_new - x1) * (1.0 if x_new > x1 else -1.0)
                
                x1, f1 = x2, f2
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                    
                if not math.isfinite(f_new):
                    break
                    
                x2 = x_new
                f2 = f_new
                
                if abs(f2) < best_abs_fx:
                    best_abs_fx = abs(f2)
                    best_x = x2
                if abs(f1) < best_abs_fx:
                    best_abs_fx = abs(f1)
                    best_x = x1
    else:
        # No bracket found. Use secant/Newton method from x0
        x_prev = x - 1e-4 * max(1.0, abs(x))
        try:
            f_prev = f(x_prev)
        except Exception:
            f_prev = fx + 1e-4  # dummy value
        
        x_curr = x
        f_curr = fx
        
        for i in range(50):
            if time.time() > deadline:
                break
                
            if abs(f_curr) <= TOL:
                best_x = x_curr
                best_abs_fx = abs(f_curr)
                break
            if abs(f_prev) <= TOL:
                best_x = x_prev
                best_abs_fx = abs(f_prev)
                break
                
            if f_curr == f_prev:
                # Try a small perturbation
                x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
                x_prev, f_prev = x_curr, f_curr
                x_curr, f_curr = x_new, f_new
                continue
                
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            if not math.isfinite(x_new):
                break
                
            # Dampen if step is too large
            if abs(x_new - x_curr) > 5.0 * max(1.0, abs(x_curr)):
                x_new = x_curr + 0.5 * (x_new - x_curr) * (1.0 if x_new > x_curr else -1.0)
            
            x_prev, f_prev = x_curr, f_curr
            x_curr = x_new
            try:
                f_curr = f(x_curr)
            except Exception:
                break
                
            if not math.isfinite(f_curr):
                # Revert
                x_curr, f_curr = x_prev, f_prev
                break
                
            if abs(f_curr) < best_abs_fx:
                best_abs_fx = abs(f_curr)
                best_x = x_curr
            if abs(f_prev) < best_abs_fx:
                best_abs_fx = abs(f_prev)
                best_x = x_prev

    return best_x

# EVOLVE-BLOCK-END
