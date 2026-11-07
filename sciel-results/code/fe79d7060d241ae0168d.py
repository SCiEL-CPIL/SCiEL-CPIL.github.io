"""Improved root finder for 3 fixed smooth polynomial equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for a smooth polynomial.
    Uses safeguarded Newton's method with bisection fallback,
    adaptive derivative estimation, and a time budget guard.
    """
    deadline = time.monotonic() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # Track best solution found so far
    best_x = x
    best_fval = abs(f(x))
    best_calls = 1
    
    def update_best(xc, fxc, calls):
        nonlocal best_x, best_fval, best_calls
        if abs(fxc) < best_fval:
            best_x = xc
            best_fval = abs(fxc)
            best_calls = calls
    
    # Check if already good enough
    if best_fval <= 1e-8:
        return best_x
    
    # Safeguarded Newton with bisection fallback
    # We'll try to establish a bracket, but if we can't, we'll do
    # pure Newton with damping and restarts.
    
    # First, try to find a bracket around x0
    # Expand interval to find sign change
    a = x
    fa = f(a)
    b = None
    fb = None
    calls = 1  # already counted fa
    
    if abs(fa) <= 1e-8:
        return a
    
    # Try expanding to find a sign change
    step = 1.0
    for i in range(50):
        if time.monotonic() > deadline:
            break
        # Try a + step
        b_candidate = a + step
        fb_candidate = f(b_candidate)
        calls += 1
        if abs(fb_candidate) <= 1e-8:
            return b_candidate
        if fa * fb_candidate < 0:
            b = b_candidate
            fb = fb_candidate
            break
        # Try a - step
        b_candidate = a - step
        fb_candidate = f(b_candidate)
        calls += 1
        if abs(fb_candidate) <= 1e-8:
            return b_candidate
        if fa * fb_candidate < 0:
            b = b_candidate
            fb = fb_candidate
            break
        step *= 2.0
        if step > 1e15:
            break
    
    # If we have a bracket, use Brent-like or bisection+Newton
    if b is not None:
        # Safeguarded Newton within bracket [a, b]
        # Ensure fa < 0 < fb or fa > 0 > fb
        if fa * fb > 0:
            # Shouldn't happen, but swap if needed
            a, b = b, a
            fa, fb = fb, fa
        
        for i in range(100):
            if time.monotonic() > deadline:
                break
            # Check convergence
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            
            # Newton step
            h = 1e-8 * max(1.0, abs(a))
            d = (f(a + h) - fa) / h
            calls += 1
            if d != 0:
                x_new = a - fa / d
            else:
                x_new = None
            
            # Check if Newton step is within bracket
            if x_new is not None and a < x_new < b:
                x = x_new
            else:
                # Bisection
                x = (a + b) / 2.0
            
            fx = f(x)
            calls += 1
            update_best(x, fx, calls)
            
            if abs(fx) <= 1e-8:
                return x
            
            # Update bracket
            if fa * fx < 0:
                b = x
                fb = fx
            else:
                a = x
                fa = fx
        
        # Return best found
        return best_x if best_fval <= 1e-8 else (a if abs(fa) < abs(fb) else b)
    
    # No bracket found: use damped Newton with restarts
    # Try multiple starting points if needed
    start_points = [x0]
    # Add some perturbed starting points
    for delta in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2]:
        start_points.append(x0 + delta)
    
    for sp in start_points:
        if time.monotonic() > deadline:
            break
        x = float(sp)
        fx = f(x)
        calls += 1
        update_best(x, fx, calls)
        
        if abs(fx) <= 1e-8:
            return x
        
        for i in range(80):
            if time.monotonic() > deadline:
                break
            if abs(fx) <= 1e-8:
                return x
            
            # Adaptive derivative
            h = 1e-8 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            calls += 1
            if d == 0:
                break
            
            # Damped Newton step
            step_size = fx / d
            # Limit step size
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step_size) > max_step:
                step_size = math.copysign(max_step, step_size)
            
            x_new = x - step_size
            fx_new = f(x_new)
            calls += 1
            update_best(x_new, fx_new, calls)
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # If function value increased significantly, try smaller step
            if abs(fx_new) > abs(fx) * 10:
                # Halve the step
                x_new = x - step_size / 2.0
                fx_new = f(x_new)
                calls += 1
                update_best(x_new, fx_new, calls)
                if abs(fx_new) <= 1e-8:
                    return x_new
                if abs(fx_new) > abs(fx):
                    # Try even smaller
                    x_new = x - step_size / 4.0
                    fx_new = f(x_new)
                    calls += 1
                    update_best(x_new, fx_new, calls)
                    if abs(fx_new) <= 1e-8:
                        return x_new
            
            x = x_new
            fx = fx_new
    
    # Final check
    if best_fval <= 1e-8:
        return best_x
    return best_x
# EVOLVE-BLOCK-END
