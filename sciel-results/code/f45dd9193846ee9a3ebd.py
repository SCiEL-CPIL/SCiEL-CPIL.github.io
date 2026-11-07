"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Hybrid root finder: efficient bracketing + bisection/secant with Newton polish.
    Designed for smooth polynomial equations. Tracks calls explicitly to stay under 500.
    """
    start_time = time.monotonic()
    deadline = start_time + 145.0  # safety margin before 150s limit
    TOL = 1e-10  # internal tolerance, stricter than required 1e-8
    MAX_CALLS = 490  # stay well under 500 per equation
    
    x0 = float(x0)
    call_count = 0
    best_x = x0
    best_f_val = float('inf')
    
    def _check_time():
        return time.monotonic() < deadline
    
    def call(x_val):
        """Call f with time guard and call counting."""
        nonlocal call_count, best_x, best_f_val
        if call_count >= MAX_CALLS:
            return None
        if not _check_time():
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            if abs(val) < best_f_val:
                best_f_val = abs(val)
                best_x = x_val
            return val
        except Exception:
            call_count += 1
            return None
    
    # Phase 1: Evaluate at x0
    fx0 = call(x0)
    if fx0 is None:
        return best_x
    if abs(fx0) <= TOL:
        return x0
    
    # Phase 2: Try to find a bracket around x0
    a = x0
    fa = fx0
    b = None
    fb = None
    
    # Try several offsets to find a bracket - use exponential expansion
    step = max(1.0, abs(x0) * 0.5)
    for direction in [-1, 1]:
        for i in range(1, 20):
            if not _check_time():
                break
            if call_count >= MAX_CALLS - 5:
                break
            x_try = x0 + direction * step * (2 ** i)
            if not math.isfinite(x_try):
                break
            fx_try = call(x_try)
            if fx_try is None:
                break
            if abs(fx_try) <= TOL:
                return x_try
            if fa * fx_try <= 0:
                if direction == -1:
                    a, fa = x_try, fx_try
                    b, fb = x0, fx0
                else:
                    a, fa = x0, fx0
                    b, fb = x_try, fx_try
                break
        if b is not None:
            break
    
    # If we found a bracket, use hybrid bisection/secant
    if b is not None and a is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Hybrid method: use secant step if it stays in bracket, else bisection
        x_prev = a
        f_prev = fa
        x_curr = b
        f_curr = fb
        
        for _ in range(80):
            if not _check_time():
                break
            if call_count >= MAX_CALLS - 5:
                break
            if abs(f_curr) <= TOL:
                return x_curr
            if abs(f_prev) <= TOL:
                return x_prev
            
            # Secant step
            if f_curr != f_prev:
                x_secant = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                x_secant = (x_curr + x_prev) / 2
            
            # Check if secant step is within bracket
            if a < x_secant < b:
                x_next = x_secant
            else:
                # Bisection
                x_next = (a + b) / 2
            
            f_next = call(x_next)
            if f_next is None:
                # Fall back to bisection
                x_next = (a + b) / 2
                f_next = call(x_next)
                if f_next is None:
                    break
            
            if abs(f_next) <= TOL:
                return x_next
            
            # Update bracket
            if fa * f_next <= 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next
            
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next
            
            # Convergence check
            if abs(b - a) < 1e-13 * max(1.0, abs(x_curr)):
                break
        
        # Final polish: Newton-like refinement using finite differences
        if abs(f_curr) <= 1e-6 and call_count < MAX_CALLS - 10:
            x = x_curr
            fx = f_curr
            for _ in range(15):
                if not _check_time():
                    break
                if call_count >= MAX_CALLS - 5:
                    break
                if abs(fx) <= TOL:
                    return x
                
                h = 1e-8 * max(1.0, abs(x))
                fxh = call(x + h)
                if fxh is None:
                    break
                
                d = (fxh - fx) / h
                if d == 0 or not math.isfinite(d):
                    break
                
                x_new = x - fx / d
                
                if not math.isfinite(x_new):
                    break
                
                # Damping if step is too large
                step_size = abs(x_new - x)
                if step_size > 5 * max(1.0, abs(x)):
                    x_new = x + math.copysign(5 * max(1.0, abs(x)), x_new - x)
                
                fx_new = call(x_new)
                if fx_new is None:
                    break
                
                if abs(fx_new) <= TOL:
                    return x_new
                
                # If new point is worse, try smaller step
                if abs(fx_new) > 2 * abs(fx):
                    x_new = x - 0.5 * fx / d
                    fx_new = call(x_new)
                    if fx_new is None:
                        break
                    if abs(fx_new) > abs(fx):
                        break
                
                x = x_new
                fx = fx_new
        
        # Return the best point found
        return best_x
    
    # Phase 3: No bracket found. Use Newton-like method with finite differences
    x = x0
    fx = fx0
    
    for _ in range(60):
        if not _check_time():
            break
        if call_count >= MAX_CALLS - 5:
            break
        if abs(fx) <= TOL:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        fxh = call(x + h)
        if fxh is None:
            break
        
        d = (fxh - fx) / h
        if d == 0 or not math.isfinite(d):
            # Try smaller step
            h = 1e-9 * max(1.0, abs(x))
            fxh = call(x + h)
            if fxh is None:
                break
            d = (fxh - fx) / h
            if d == 0 or not math.isfinite(d):
                break
        
        x_new = x - fx / d
        
        # Damping if step is too large
        if not math.isfinite(x_new):
            break
        step_size = abs(x_new - x)
        if step_size > 10 * max(1.0, abs(x)):
            x_new = x + math.copysign(10 * max(1.0, abs(x)), x_new - x)
        
        fx_new = call(x_new)
        if fx_new is None:
            break
        
        # If new point is worse, try smaller step
        if abs(fx_new) > 2 * abs(fx):
            x_new = x - 0.5 * fx / d
            fx_new = call(x_new)
            if fx_new is None:
                break
            if abs(fx_new) > abs(fx):
                break
        
        x = x_new
        fx = fx_new
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
