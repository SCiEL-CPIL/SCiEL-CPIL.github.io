"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free root finder using true inverse quadratic interpolation (IQI)
    with safeguarding, targeting smooth polynomial equations.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if abs(fx) <= 1e-12:
        return x
    
    # We'll maintain a history of (x, f(x)) points for IQI
    # Start with 2 points to bootstrap
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-12:
        return x1
    
    # Now we have 2 points: (x, fx), (x1, f1)
    # For IQI we need 3 points. Get a third point.
    
    history = [(x, fx), (x1, f1)]
    best_x = x if abs(fx) < abs(f1) else x1
    best_fx = min(abs(fx), abs(f1))
    
    # Try to get a third point that ideally brackets
    if fx * f1 < 0:
        # We have a bracket [x, x1]. Good.
        # Do one secant step to get a third point inside.
        x_new = x - fx * (x1 - x) / (f1 - fx)
        # Safeguard: keep within bracket
        if not (min(x, x1) < x_new < max(x, x1)):
            x_new = (x + x1) / 2.0
        f_new = f(x_new)
        if abs(f_new) <= 1e-12:
            return x_new
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        # Update bracket
        if fx * f_new < 0:
            x1 = x_new
            f1 = f_new
        else:
            x = x_new
            fx = f_new
        history.append((x_new, f_new))
    else:
        # No bracket. Try a larger step.
        h2 = 1e-2 * max(1.0, abs(x))
        x2 = x + h2
        f2 = f(x2)
        if abs(f2) <= 1e-12:
            return x2
        if abs(f2) < best_fx:
            best_fx = abs(f2)
            best_x = x2
        # Pick the best 3 points for history
        history = [(x, fx), (x1, f1), (x2, f2)]
    
    # Main iteration loop
    # We'll use true IQI (fit quadratic in f-space) with secant fallback
    
    for it in range(50):
        if time.time() > deadline:
            break
        if len(history) < 3:
            # Need more points
            # Use secant on best 2
            (xa, fa), (xb, fb) = history[-2], history[-1]
            if fa == fb:
                break
            x_new = xa - fa * (xb - xa) / (fb - fa)
            f_new = f(x_new)
            if abs(f_new) <= 1e-12:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            history.append((x_new, f_new))
            if len(history) > 5:
                history.pop(0)
            continue
        
        # True IQI: fit quadratic p(f) through (f0, x0), (f1, x1), (f2, x2)
        # and find f_new such that p(f_new) = 0 (i.e., the root of the inverse quadratic)
        (x0v, f0v), (x1v, f1v), (x2v, f2v) = history[-3], history[-2], history[-1]
        
        x_new = None
        if f0v != f1v and f1v != f2v and f0v != f2v:
            # True IQI formula:
            # x_new = x0 * f1 * f2 / ((f0 - f1) * (f0 - f2))
            #       + x1 * f0 * f2 / ((f1 - f0) * (f1 - f2))
            #       + x2 * f0 * f1 / ((f2 - f0) * (f2 - f1))
            term0 = x0v * f1v * f2v / ((f0v - f1v) * (f0v - f2v))
            term1 = x1v * f0v * f2v / ((f1v - f0v) * (f1v - f2v))
            term2 = x2v * f0v * f1v / ((f2v - f0v) * (f2v - f1v))
            x_new = term0 + term1 + term2
        
        # Fallback to secant if IQI is degenerate or non-finite
        if x_new is None or not math.isfinite(x_new):
            (xa, fa), (xb, fb) = history[-2], history[-1]
            if fa == fb:
                break
            x_new = xa - fa * (xb - xa) / (fb - fa)
        
        # Safeguard: don't go too far from current points
        if not math.isfinite(x_new):
            break
        
        # If x_new is very far from the current points, use secant instead
        x_ref = history[-1][0]
        x_scale = max(1.0, abs(x_ref))
        if abs(x_new - x_ref) > 10.0 * x_scale:
            (xa, fa), (xb, fb) = history[-2], history[-1]
            if fa != fb:
                x_new = xa - fa * (xb - xa) / (fb - fa)
        
        f_new = f(x_new)
        if abs(f_new) <= 1e-12:
            return x_new
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        
        # Update history: keep the 3 most recent points
        history.append((x_new, f_new))
        if len(history) > 3:
            history.pop(0)
        
        # Check for convergence
        if abs(f_new) <= 1e-10:
            return x_new
    
    # Final high-precision polish: if we have a good candidate, try Newton-like
    # refinement using finite differences to get below 1e-8 more reliably
    if best_fx > 1e-8 and best_fx < 1e-6:
        # Try a few secant steps from the best point
        xp = best_x
        fp = f(xp)
        for _ in range(10):
            if time.time() > deadline:
                break
            if abs(fp) <= 1e-8:
                return xp
            # Estimate derivative
            dx = 1e-8 * max(1.0, abs(xp))
            fp2 = f(xp + dx)
            df = (fp2 - fp) / dx
            if abs(df) < 1e-30:
                break
            x_new = xp - fp / df
            if not math.isfinite(x_new):
                break
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            xp = x_new
            fp = f_new
    
    return best_x
# EVOLVE-BLOCK-END
