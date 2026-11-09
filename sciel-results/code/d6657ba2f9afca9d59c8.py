"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find a root of f near x0 using a combination of:
    1. Adaptive geometric expansion to find a bracket [a, b] where f(a)*f(b) < 0
    2. Bisection with secant/Newton acceleration for fast convergence
    3. Deadline guard to ensure we finish within time limits
    """
    tol = 1e-8
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    x = float(x0)
    
    # Count calls via a wrapper
    call_count = [0]
    def eval_f(x):
        call_count[0] += 1
        return f(x)
    
    # First evaluation
    fx = eval_f(x)
    if not math.isfinite(fx):
        # If f(x0) is non-finite, try small perturbations
        for delta in [1e-6, 1e-4, 1e-2, 1.0]:
            for sign in [1, -1]:
                xp = x + sign * delta
                fp = eval_f(xp)
                if math.isfinite(fp) and abs(fp) <= tol:
                    return xp
        return x  # Best effort
    
    if abs(fx) <= tol:
        return x
    
    # Phase 1: Find a bracket [a, b] with f(a)*f(b) < 0
    # Use adaptive geometric expansion
    a = x
    fa = fx
    
    # Try small steps first, then expand geometrically
    step_sizes = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
    
    b = None
    fb = None
    
    for delta in step_sizes:
        # Check both directions
        for sign in [1, -1]:
            if time.time() > deadline:
                break
            xp = x + sign * delta
            fp = eval_f(xp)
            if not math.isfinite(fp):
                continue
            if abs(fp) <= tol:
                return xp
            if fa * fp < 0:
                b = xp
                fb = fp
                break
        if b is not None:
            break
        if time.time() > deadline:
            break
    
    # If we found a bracket, use bisection + secant acceleration
    if b is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Bisection with secant/Newton acceleration
        for _ in range(200):
            if time.time() > deadline:
                break
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            if not math.isfinite(fm):
                # Fallback to pure bisection point
                mid = (a + b) / 2.0
                fm = eval_f(mid)
            
            if abs(fm) <= tol:
                return mid
            
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            
            # Try secant step for acceleration
            if a != b and fa != fb and fa * fb < 0:
                # Secant method estimate
                denom = fb - fa
                if abs(denom) > 1e-30:
                    x_sec = b - fb * (b - a) / denom
                    # Only accept if it's within the bracket
                    if a < x_sec < b:
                        x_sec_f = eval_f(x_sec)
                        if math.isfinite(x_sec_f):
                            if abs(x_sec_f) <= tol:
                                return x_sec
                            # Update bracket
                            if fa * x_sec_f <= 0:
                                b = x_sec
                                fb = x_sec_f
                            else:
                                a = x_sec
                                fa = x_sec_f
            
            # Check convergence
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                return mid
        
        # Return the midpoint of the final bracket
        return (a + b) / 2.0
    
    # Phase 2: No bracket found. Try Newton-like method with finite differences
    # as a fallback
    x_curr = x
    for _ in range(100):
        if time.time() > deadline:
            break
        fc = eval_f(x_curr)
        if not math.isfinite(fc):
            break
        if abs(fc) <= tol:
            return x_curr
        
        h = 1e-7 * max(1.0, abs(x_curr))
        d = (eval_f(x_curr + h) - fc) / h
        if not math.isfinite(d) or abs(d) < 1e-30:
            break
        x_new = x_curr - fc / d
        if not math.isfinite(x_new):
            break
        # Damping if step is too large
        if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
            x_new = x_curr + 10 * max(1.0, abs(x_curr)) * math.copysign(1, x_new - x_curr)
        x_curr = x_new
    
    return x_curr

# EVOLVE-BLOCK-END
