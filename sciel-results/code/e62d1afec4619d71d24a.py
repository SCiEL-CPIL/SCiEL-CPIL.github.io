"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free root finder using pure secant method with safeguarding,
    targeting smooth polynomial equations.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if abs(fx) <= 1e-12:
        return x
    
    # Start with 2 points for secant method
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-12:
        return x1
    
    best_x = x if abs(fx) < abs(f1) else x1
    best_fx = min(abs(fx), abs(f1))
    
    # Main secant iteration loop
    for it in range(30):
        if time.time() > deadline:
            break
        
        # Secant step
        if fx == f1:
            break
        x_new = x - fx * (x1 - x) / (f1 - fx)
        
        if not math.isfinite(x_new):
            break
        
        # Safeguard: don't go too far from current points
        x_ref = x1
        x_scale = max(1.0, abs(x_ref))
        if abs(x_new - x_ref) > 10.0 * x_scale:
            # Use bisection-like step instead
            x_new = (x + x1) / 2.0
        
        f_new = f(x_new)
        if abs(f_new) <= 1e-12:
            return x_new
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        
        # Update: keep the two most recent points
        x, fx = x1, f1
        x1, f1 = x_new, f_new
        
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
