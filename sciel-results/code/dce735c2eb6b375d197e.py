"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 110.0  # 110s safety margin before 120s kill
    
    def time_up():
        return time.time() > deadline
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return x
    
    # Phase 1: Secant method (1 new eval per iter after initial 2)
    # Start with a small perturbation
    h = 1e-4 * max(1.0, abs(x))
    x_prev = x
    x_curr = x + h
    fx_prev = best_fx
    fx_curr = f(x_curr)
    
    if abs(fx_curr) < best_abs:
        best_x, best_fx, best_abs = x_curr, fx_curr, abs(fx_curr)
        if best_abs <= 1e-8:
            return x_curr
    
    # Secant iterations
    for i in range(30):
        if time_up():
            break
        if fx_prev == fx_curr:
            break
        denom = fx_curr - fx_prev
        if abs(denom) < 1e-30:
            break
        x_next = x_curr - fx_curr * (x_curr - x_prev) / denom
        
        # Check for non-finite
        if not math.isfinite(x_next):
            break
        
        fx_next = f(x_next)
        if not math.isfinite(fx_next):
            break
        
        if abs(fx_next) < best_abs:
            best_x, best_fx, best_abs = x_next, fx_next, abs(fx_next)
        
        if best_abs <= 1e-8:
            return x_next
        
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_next, fx_next
    
    if time_up():
        return best_x
    
    # Phase 2: Find a bracket and use Brent's method
    # Expand from current best position
    a = best_x - 1.0
    b = best_x + 1.0
    fa = f(a)
    fb = f(b)
    
    # Try to find a sign change by expanding
    found_bracket = False
    if math.isfinite(fa) and math.isfinite(fb) and fa * fb <= 0:
        found_bracket = True
    else:
        # Expand outward
        step = 1.0
        for _ in range(50):
            if time_up():
                break
            a_new = best_x - step
            b_new = best_x + step
            fa = f(a_new)
            fb = f(b_new)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                step *= 2.0
                continue
            if fa * fb <= 0:
                a, b = a_new, b_new
                found_bracket = True
                break
            step *= 2.0
            if step > 1e12:
                break
    
    if found_bracket and not time_up():
        try:
            from scipy.optimize import brentq
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=50)
            if math.isfinite(root):
                return root
        except Exception:
            pass
    
    # Phase 3: Fallback - bisection if we have a bracket, otherwise return best
    if found_bracket and not time_up():
        # Manual bisection as fallback
        for _ in range(60):
            if time_up():
                break
            mid = 0.5 * (a + b)
            fm = f(mid)
            if not math.isfinite(fm):
                break
            if abs(fm) < best_abs:
                best_x, best_fx, best_abs = mid, fm, abs(fm)
            if best_abs <= 1e-8:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
    
    return best_x
# EVOLVE-BLOCK-END
