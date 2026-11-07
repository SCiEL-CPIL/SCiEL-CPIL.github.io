"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free root finder using safeguarded secant with adaptive
    safeguarding, plus a final high-precision polish pass.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if not math.isfinite(fx):
        # Try a perturbation
        h = 1e-4 * max(1.0, abs(x))
        x1 = x + h
        f1 = f(x1)
        if not math.isfinite(f1):
            return x
        fx, x = f1, x1
    
    if abs(fx) <= 1e-12:
        return x
    
    # Start with 2 points for secant method
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = f(x1)
    
    if not math.isfinite(f1):
        # Try other perturbation
        x1 = x - h
        f1 = f(x1)
        if not math.isfinite(f1):
            return x
    
    if abs(f1) <= 1e-12:
        return x1
    
    best_x = x if abs(fx) < abs(f1) else x1
    best_fx = min(abs(fx), abs(f1))
    
    # Track call count
    call_count = 3  # x, x+h, and possibly x-h
    
    # Main secant iteration loop with adaptive safeguarding
    # Use a more aggressive safeguard: allow up to 5x the bracket width
    safeguard_factor = 5.0
    for it in range(40):
        if time.time() > deadline:
            break
        
        # Secant step
        denom = f1 - fx
        if abs(denom) < 1e-30:
            # Degenerate: use bisection-like step
            x_new = (x + x1) / 2.0
        else:
            x_new = x - fx * (x1 - x) / denom
        
        if not math.isfinite(x_new):
            # Fallback to midpoint
            x_new = (x + x1) / 2.0
        
        # Safeguard: don't go too far from current points
        x_ref = (x + x1) / 2.0
        x_scale = max(1.0, abs(x_ref))
        if abs(x_new - x_ref) > safeguard_factor * x_scale:
            # Use bisection-like step instead
            x_new = (x + x1) / 2.0
        
        f_new = f(x_new)
        call_count += 1
        
        if not math.isfinite(f_new):
            # Revert: keep old points, shrink step
            x_new = (x + x1) / 2.0
            f_new = f(x_new)
            call_count += 1
            if not math.isfinite(f_new):
                break
        
        if abs(f_new) <= 1e-12:
            return x_new
        
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        
        # Check for convergence
        if abs(f_new) <= 1e-10:
            return x_new
        
        # Update: keep the two most recent points
        x, fx = x1, f1
        x1, f1 = x_new, f_new
    
    # Final high-precision polish: if we have a good candidate, try Newton-like
    # refinement using finite differences to get below 1e-8 more reliably
    if best_fx > 1e-8 and best_fx < 1e-4:
        # Try a few secant steps from the best point
        xp = best_x
        fp = f(xp)
        for _ in range(15):
            if time.time() > deadline:
                break
            if abs(fp) <= 1e-8:
                return xp
            # Estimate derivative with central difference for better accuracy
            dx = 1e-6 * max(1.0, abs(xp))
            fp_plus = f(xp + dx)
            fp_minus = f(xp - dx)
            call_count += 2
            if not math.isfinite(fp_plus) or not math.isfinite(fp_minus):
                break
            df = (fp_plus - fp_minus) / (2 * dx)
            if abs(df) < 1e-30:
                break
            x_new = xp - fp / df
            if not math.isfinite(x_new):
                break
            f_new = f(x_new)
            call_count += 1
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            xp = x_new
            fp = f_new
    
    return best_x
# EVOLVE-BLOCK-END
