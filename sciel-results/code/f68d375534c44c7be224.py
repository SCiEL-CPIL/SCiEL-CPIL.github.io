"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def find_root(f, x0):
    TOL = 1e-8
    POLISH_TOL = 1e-12
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0  # safety margin before 150s limit
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS:
            return None
        if time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isnan(val) or math.isinf(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
        return val
    
    def is_good(val):
        return val is not None and abs(val) <= TOL
    
    def is_polish_good(val):
        return val is not None and abs(val) <= POLISH_TOL
    
    def newton_raphson(x0, max_iter=60, use_forward_diff=True):
        """Newton with adaptive derivative estimation. Forward diff saves evals."""
        nonlocal evals, best_x, best_abs
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return best_x
        if is_good(fx):
            return x
        
        prev_x = x
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE:
                return best_x
            
            # Adaptive step size for derivative
            h = 1e-7 * max(1.0, abs(x))
            
            if use_forward_diff:
                xp = x + h
                fpx = call_f(xp)
                if fpx is None:
                    # Try central as fallback
                    xm = x - h
                    fxm = call_f(xm)
                    if fxm is not None:
                        d = (fpx if fpx is not None else fx - fxm) / h if fpx is None else (fx - fxm) / h
                        if fpx is None:
                            d = (fx - fxm) / h
                    else:
                        break
                else:
                    d = (fpx - fx) / h
            else:
                xp = x + h
                xm = x - h
                fpx = call_f(xp)
                fxm = call_f(xm)
                
                if fpx is not None and fxm is not None:
                    d = (fpx - fxm) / (2 * h)
                elif fpx is not None:
                    d = (fpx - fx) / h
                elif fxm is not None:
                    d = (fx - fxm) / h
                else:
                    break
            
            if d is None or d == 0 or math.isnan(d) or math.isinf(d):
                break
            
            x_new = x - fx / d
            
            # Safeguard: if step is too large, damp it
            max_step = 5.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + 0.5 * (x_new - x)
            
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            
            if is_good(fx_new):
                return x_new
            
            # Check for convergence or stagnation
            if abs(x_new - x) < 1e-13 * max(1.0, abs(x_new)):
                break
            
            # If we're getting worse, revert and try smaller step
            if abs(fx_new) > abs(fx) * 1.1:
                # Try a damped step
                x_new = x - 0.5 * fx / d
                fx_new = call_f(x_new)
                if fx_new is None:
                    break
                if is_good(fx_new):
                    return x_new
            
            x = x_new
            fx = fx_new
        
        return best_x
    
    def secant(x0, x1, max_iter=60):
        nonlocal evals, best_x, best_abs
        x_prev = float(x0)
        x_curr = float(x1)
        f_prev = call_f(x_prev)
        f_curr = call_f(x_curr)
        
        if f_prev is None or f_curr is None:
            return best_x
        if is_good(f_prev):
            return x_prev
        if is_good(f_curr):
            return x_curr
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE:
                return best_x
            
            denom = f_curr - f_prev
            if denom == 0 or math.isnan(denom) or math.isinf(denom):
                break
            
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            
            # Damping
            if abs(x_new - x_curr) > 5.0 * max(1.0, abs(x_curr)):
                x_new = x_curr + 0.5 * (x_new - x_curr)
            
            f_new = call_f(x_new)
            if f_new is None:
                break
            
            if is_good(f_new):
                return x_new
            
            if abs(x_new - x_curr) < 1e-13 * max(1.0, abs(x_new)):
                break
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new
        
        return best_x
    
    def bisection(xa, xb, fa, fb, max_iter=80):
        nonlocal evals, best_x, best_abs
        if fa is None or fb is None:
            return best_x
        if is_good(fa):
            return xa
        if is_good(fb):
            return xb
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            return best_x
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE:
                return best_x
            
            x_mid = (xa + xb) / 2.0
            f_mid = call_f(x_mid)
            if f_mid is None:
                break
            
            if is_good(f_mid):
                return x_mid
            
            if f_mid * fa <= 0:
                xb = x_mid
                fb = f_mid
            else:
                xa = x_mid
                fa = f_mid
            
            if abs(xb - xa) < 1e-13 * max(1.0, abs(x_mid)):
                break
        
        return best_x
    
    def polish(x_start, max_iter=20):
        """High-precision polish: use secant with very tight tolerance."""
        nonlocal evals, best_x, best_abs
        x = float(x_start)
        fx = call_f(x)
        if fx is None:
            return best_x
        if is_good(fx):
            return x
        
        h = 1e-10 * max(1.0, abs(x))
        x_prev = x - h
        x_curr = x
        f_prev = call_f(x_prev)
        f_curr = fx
        
        if f_prev is None:
            return best_x
        if is_good(f_prev):
            return x_prev
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE:
                return best_x
            
            denom = f_curr - f_prev
            if denom == 0 or math.isnan(denom) or math.isinf(denom):
                break
            
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            
            if abs(x_new - x_curr) > 1e-6 * max(1.0, abs(x_curr)):
                x_new = x_curr + 0.5 * (x_new - x_curr)
            
            f_new = call_f(x_new)
            if f_new is None:
                break
            
            if is_good(f_new):
                return x_new
            
            if abs(f_new) < 1e-14:
                return x_new
            
            if abs(x_new - x_curr) < 1e-15 * max(1.0, abs(x_new)):
                break
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new
        
        return best_x
    
    # Strategy: Multi-phase approach with safeguarding
    x0 = float(x0)
    
    # Phase 1: Newton's method with forward differences (cheaper per iteration)
    result = newton_raphson(x0, max_iter=40, use_forward_diff=True)
    if best_abs <= TOL:
        # Polish for extra precision
        result = polish(best_x, max_iter=15)
        return best_x
    
    # Phase 2: Secant method with perturbed starting points
    h = 1e-4 * max(1.0, abs(x0))
    result = secant(x0, x0 + h, max_iter=40)
    if best_abs <= TOL:
        result = polish(best_x, max_iter=15)
        return best_x
    
    result = secant(x0 - h, x0 + h, max_iter=40)
    if best_abs <= TOL:
        result = polish(best_x, max_iter=15)
        return best_x
    
    # Phase 3: Try to find a bracket and use bisection
    step = 1e-2 * max(1.0, abs(x0))
    for i in range(1, 15):
        if time.monotonic() > DEADLINE:
            break
        if evals >= MAX_EVALS - 30:
            break
        
        xa = x0 - i * step
        xb = x0 + i * step
        fa = call_f(xa)
        fb = call_f(xb)
        
        if fa is None or fb is None:
            continue
        
        if is_good(fa):
            return xa
        if is_good(fb):
            return xb
        
        if fa * fb < 0:
            result = bisection(xa, xb, fa, fb, max_iter=50)
            if best_abs <= TOL:
                result = polish(best_x, max_iter=15)
                return best_x
    
    # Phase 4: Random restarts if we have budget remaining
    if evals < MAX_EVALS - 20 and time.monotonic() < DEADLINE - 5:
        rng = random.Random(42)
        for _ in range(5):
            if evals >= MAX_EVALS - 10 or time.monotonic() > DEADLINE - 3:
                break
            # Random perturbation around best_x
            scale = max(1.0, abs(best_x)) * 0.1
            x_try = best_x + rng.uniform(-scale, scale)
            result = newton_raphson(x_try, max_iter=20, use_forward_diff=True)
            if best_abs <= TOL:
                result = polish(best_x, max_iter=10)
                return best_x
    
    # Final polish on best found
    if best_abs < 1e-4:
        result = polish(best_x, max_iter=10)
    
    return best_x
# EVOLVE-BLOCK-END
