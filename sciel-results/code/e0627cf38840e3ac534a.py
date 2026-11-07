"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # Best point tracking
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # 2. Initialize Secant Method
    # We need a second point to start the secant method.
    # Choose a step size h that is small enough to be in the local region 
    # but large enough to avoid numerical noise.
    # For polynomials, a relative step is usually good.
    h = max(1e-4, abs(x) * 1e-4)
    
    x_prev = x
    f_prev = fx
    
    x_curr = x + h
    try:
        f_curr = f(x_curr)
    except Exception:
        # If the first step fails, try a different step
        x_curr = x - h
        try:
            f_curr = f(x_curr)
        except Exception:
            return best_x
        
    if not math.isfinite(f_curr):
        # If non-finite, try a few more points
        found = False
        for dx in [1e-3, 1e-2, 0.1, 1.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x_curr = x_test
                    f_curr = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return best_x

    if abs(f_curr) <= 1e-8:
        return x_curr
    if abs(f_curr) < best_abs_fx:
        best_x = x_curr
        best_fx = f_curr
        best_abs_fx = abs(f_curr)

    # Secant Iteration
    # Convergence order is ~1.618. 
    # Max iterations should be low to save calls if it fails.
    max_iter = 30
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check for zero slope
        if abs(f_curr - f_prev) < 1e-14:
            # Perturb to escape
            # Try a small random step or a fixed step
            x_curr = x_curr + 1e-3
            try:
                f_curr = f(x_curr)
            except Exception:
                break
            if not math.isfinite(f_curr):
                break
            if abs(f_curr) < best_abs_fx:
                best_x = x_curr
                best_fx = f_curr
                best_abs_fx = abs(f_curr)
            if abs(f_curr) <= 1e-8:
                return x_curr
            continue

        # Secant step
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # Safeguard: If step is too large, shrink it
        # A good heuristic for root finding is to ensure the step doesn't 
        # jump too far from the current best or current point.
        max_step = max(1.0, abs(x_curr) * 0.5)
        if abs(x_new - x_curr) > max_step:
            # Shrink towards x_curr
            x_new = x_curr + math.copysign(max_step, x_new - x_curr)
            
        # Evaluate new point
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(f_new):
            # If non-finite, try a smaller step (bisection-like)
            x_new = 0.5 * (x_curr + x_prev)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        # Update best
        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_fx = f_new
            best_abs_fx = abs(f_new)
            
        if abs(f_new) <= 1e-8:
            return x_new

        # Update history
        # Standard secant moves: x_prev = x_curr, x_curr = x_new
        # However, if the step was large and shrunk, or if we are oscillating,
        # we might want to keep the best point. 
        # Standard secant is usually fine for polynomials.
        
        # Stagnation check: if x_new is very close to x_curr or x_prev, 
        # we might be stuck.
        if abs(x_new - x_curr) < 1e-12 * max(1.0, abs(x_curr)):
             # Perturb
             x_new = x_curr + 1e-4
             try:
                 f_new = f(x_new)
             except Exception:
                 break
             if not math.isfinite(f_new):
                 break
             if abs(f_new) < best_abs_fx:
                 best_x = x_new
                 best_fx = f_new
                 best_abs_fx = abs(f_new)
             if abs(f_new) <= 1e-8:
                 return x_new

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new

    return best_x
# EVOLVE-BLOCK-END
