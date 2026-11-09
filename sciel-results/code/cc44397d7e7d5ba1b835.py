"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Secant method with adaptive safeguards and a final Newton polish.
    # Track elapsed time to stay well under the 150s kill deadline.
    deadline = time.time() + 140.0  # safety margin
    
    x = float(x0)
    fx = f(x)
    
    # If we're already at the root, return immediately
    if abs(fx) <= 1e-8:
        return x
    
    # Initialize secant: need two points. Use a small perturbation for the second.
    h = 1e-4 * max(1.0, abs(x))
    x_prev = x + h
    fx_prev = f(x_prev)
    
    # If the perturbation gave us a better point, use it as our best
    best_x = x
    best_fx = fx
    if abs(fx_prev) < abs(best_fx):
        best_x = x_prev
        best_fx = fx_prev
        # Swap so x is the more recent, x_prev is older
        x, x_prev = x_prev, x
        fx, fx_prev = fx_prev, fx
    
    # Secant iteration
    for i in range(40):
        if time.time() > deadline:
            return best_x
        
        # Check convergence
        if abs(fx) <= 1e-8:
            # Do a quick Newton polish if derivative is available
            return _polish(f, x, fx, deadline)
        
        # Update best
        if abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        
        # Compute secant step
        if fx_prev == fx:
            # Degenerate case: try a different perturbation
            h_new = 1e-4 * max(1.0, abs(x)) * (1.0 if i % 2 == 0 else -1.0)
            x_new = x + h_new
            fx_new = f(x_new)
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            # Reset secant with new point
            x_prev, fx_prev = x_new, fx_new
            x, fx = x_new, fx_new
            continue
        
        denom = fx - fx_prev
        if denom == 0:
            break
        
        dx = (x - x_prev) * fx / denom
        x_new = x - dx
        
        # Safeguard: limit step size to prevent divergence
        max_step = 10.0 * max(1.0, abs(x))
        if abs(dx) > max_step:
            x_new = x + math.copysign(max_step, dx)
        
        fx_new = f(x_new)
        
        # If new point is not finite, try a smaller step
        if not math.isfinite(fx_new):
            x_new = x - 0.5 * dx
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                return best_x
        
        # Update secant pair: keep the two most recent distinct points
        x_prev, fx_prev = x, fx
        x, fx = x_new, fx_new
    
    # Final check
    if abs(fx) < abs(best_fx):
        best_x = x
        best_fx = fx
    
    # Polish the best point found
    return _polish(f, best_x, best_fx, deadline)


def _polish(f, x, fx, deadline):
    """Newton-Raphson polish with finite-difference derivative."""
    if abs(fx) <= 1e-12:
        return x
    
    for _ in range(10):
        if time.time() > deadline:
            return x
        
        h = 1e-8 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
        
        fx_new = f(x_new)
        
        if not math.isfinite(fx_new):
            break
        
        if abs(fx_new) >= abs(fx):
            # Step made things worse; try half step
            x_half = x - 0.5 * fx / d
            fx_half = f(x_half)
            if math.isfinite(fx_half) and abs(fx_half) < abs(fx):
                x, fx = x_half, fx_half
            else:
                break
        else:
            x, fx = x_new, fx_new
        
        if abs(fx) <= 1e-12:
            break
    
    return x
# EVOLVE-BLOCK-END
