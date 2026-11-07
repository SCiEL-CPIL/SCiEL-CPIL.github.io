# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses an efficient Secant method with Newton fallbacks.
    """
    import math
    import time
    
    DEADLINE = 110.0
    START_TIME = time.time()
    
    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    def is_finite(x):
        return math.isfinite(x)

    def safe_f(x):
        try:
            fx = f(x)
            if is_finite(fx):
                return fx
        except:
            pass
        return None

    # Initial state
    x = float(x0)
    fx = safe_f(x)
    
    if fx is None:
        return x # Failure case, return x0

    if abs(fx) <= 1e-8:
        return x

    # Set up Secant method
    # Choose initial step size
    delta = 1e-4 * max(1.0, abs(x))
    x_prev = x
    x_curr = x + delta
    
    fx_prev = fx
    fx_curr = safe_f(x_curr)
    
    if fx_curr is None:
        # Fallback to Newton if second point fails? Or just return best so far.
        return x if abs(fx) <= 1e-8 else x

    if abs(fx_curr) <= 1e-8:
        return x_curr

    # Main loop: Secant Method
    # We track best_x to return if we fail to converge
    best_x = x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
    best_fx = fx_curr if abs(fx_curr) < abs(fx_prev) else fx_prev

    for _ in range(100):
        if not check_deadline():
            break
            
        # Check convergence
        if abs(fx_curr) <= 1e-8:
            return x_curr
        if abs(fx_prev) <= 1e-8:
            return x_prev

        # Update best
        if abs(fx_curr) < abs(best_fx):
            best_x = x_curr
            best_fx = fx_curr
        if abs(fx_prev) < abs(best_fx):
            best_x = x_prev
            best_fx = fx_prev

        if fx_prev == fx_curr:
            # Parallel lines, try Newton step from best point
            break

        # Secant step
        denom = fx_curr - fx_prev
        if denom == 0 or not is_finite(denom):
            break
            
        x_next = x_curr - fx_curr * (x_curr - x_prev) / denom
        
        if not is_finite(x_next):
            break
            
        # Safeguard: If step is too large, dampen it
        # Heuristic: limit step to 10x distance from origin or 10x previous step
        max_step = 10.0 * max(1.0, abs(x_curr))
        if abs(x_next - x_curr) > max_step:
            # Dampen
            direction = 1 if x_next > x_curr else -1
            x_next = x_curr + direction * max_step
            
        fx_next = safe_f(x_next)
        
        if fx_next is None:
            # Try a smaller step or Newton
            break
            
        # Shift
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = fx_next

    # If Secant failed or stalled, try Newton from best_x
    if check_deadline() and abs(best_fx) > 1e-8:
        x = best_x
        fx = best_fx
        
        for _ in range(50):
            if not check_deadline():
                break
                
            if abs(fx) <= 1e-8:
                return x
                
            # Central difference for derivative
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = safe_f(x + h)
            fx_minus = safe_f(x - h)
            
            if fx_plus is None or fx_minus is None:
                break
                
            d = (fx_plus - fx_minus) / (2 * h)
            
            if d == 0 or not is_finite(d):
                break
                
            step = fx / d
            if not is_finite(step):
                break
                
            # Damp large steps
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = max_step * (1 if step > 0 else -1)
                
            x_new = x - step
            if not is_finite(x_new):
                break
                
            fx_new = safe_f(x_new)
            if fx_new is None:
                break
                
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Step was bad, try smaller step? Or break.
                # For robustness, just break and return best.
                break

    return best_x
# EVOLVE-BLOCK-END
