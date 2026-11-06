import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses Illinois method with adaptive step-doubling for bracketing.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit
    max_evals = 500
    eval_count = 0
    
    def check_time():
        return time.time() > deadline

    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        try:
            fx = f(x)
            eval_count += 1
            if math.isnan(fx) or math.isinf(fx):
                return None
            return fx
        except Exception:
            return None

    # Initial evaluation
    x0 = float(x0)
    fx0 = eval_f(x0)
    if fx0 is None:
        return x0
        
    best_x = x0
    best_fx = fx0
    
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy: Find a bracket using adaptive step-doubling, then use Illinois method
    # Illinois method is a modified bisection that converges faster for ill-conditioned problems
    
    # Try to find a bracket by expanding intervals around x0
    scale = max(1.0, abs(x0))
    delta = 1e-4 * scale
    
    # Points to track for Illinois method
    a, b = None, None
    fa, fb = None, None
    bracket_found = False
    
    # Adaptive step-doubling to find bracket
    max_doublings = 20
    for _ in range(max_doublings):
        if check_time() or eval_count >= max_evals - 10:
            break
            
        # Try right
        x_r = x0 + delta
        fx_r = eval_f(x_r)
        if fx_r is not None:
            if abs(fx_r) < abs(best_fx):
                best_x, best_fx = x_r, fx_r
            if abs(fx_r) <= 1e-8:
                return x_r
            if fx0 * fx_r < 0:
                a, b = x0, x_r
                fa, fb = fx0, fx_r
                bracket_found = True
                break
        
        if check_time() or eval_count >= max_evals - 10:
            break
            
        # Try left
        x_l = x0 - delta
        fx_l = eval_f(x_l)
        if fx_l is not None:
            if abs(fx_l) < abs(best_fx):
                best_x, best_fx = x_l, fx_l
            if abs(fx_l) <= 1e-8:
                return x_l
            if fx0 * fx_l < 0:
                a, b = x_l, x0
                fa, fb = fx_l, fx0
                bracket_found = True
                break
        
        # Check if fx_r and fx_l bracket
        if fx_r is not None and fx_l is not None and fx_r * fx_l < 0:
            a, b = x_l, x_r
            fa, fb = fx_l, fx_r
            bracket_found = True
            break
            
        # Double the step size
        delta *= 2.0

    # If we found a bracket, use Illinois method
    if bracket_found:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Illinois method: modified bisection with faster convergence
        # Shrink the larger subinterval by factor 0.5 instead of standard bisection
        x_illinois = (a + b) / 2.0
        fx_illinois = eval_f(x_illinois)
        
        if fx_illinois is None:
            return best_x
            
        if abs(fx_illinois) <= 1e-8:
            return x_illinois
            
        # Illinois iteration
        for _ in range(50):  # Max 50 iterations of Illinois
            if check_time() or eval_count >= max_evals - 5:
                break
                
            if fa * fx_illinois <= 0:
                b = x_illinois
                fb = fx_illinois
            else:
                a = x_illinois
                fa = fx_illinois
                
            # Illinois update: shrink larger subinterval by 0.5
            # Instead of midpoint, use weighted average
            if abs(b - x_illinois) > abs(x_illinois - a):
                # b is farther, shrink toward x_illinois
                x_illinois = x_illinois + 0.5 * (b - x_illinois)
            else:
                # a is farther, shrink toward x_illinois
                x_illinois = x_illinois + 0.5 * (a - x_illinois)
                
            if abs(x_illinois - best_x) < 1e-15:
                break
                
            fx_illinois = eval_f(x_illinois)
            if fx_illinois is None:
                break
            if abs(fx_illinois) < abs(best_fx):
                best_x, best_fx = x_illinois, fx_illinois
            if abs(fx_illinois) <= 1e-8:
                return x_illinois

        # Verify final result
        if fx_illinois is not None and abs(fx_illinois) <= 1e-8:
            return x_illinois
        return best_x

    # If no bracket found, try secant method as fallback
    # Use two points to estimate derivative
    if eval_count < max_evals - 20:
        x1 = x0 + 1e-6 * scale
        fx1 = eval_f(x1)
        if fx1 is not None and fx0 is not None and fx0 != fx1:
            # Secant iteration
            x_secant = x0
            fx_secant = fx0
            x_prev = x1
            fx_prev = fx1
            
            for _ in range(30):
                if check_time() or eval_count >= max_evals - 10:
                    break
                    
                if abs(fx_prev - fx_secant) < 1e-15:
                    break
                    
                x_new = x_secant - fx_secant * (x_prev - x_secant) / (fx_prev - fx_secant)
                fx_new = eval_f(x_new)
                
                if fx_new is None:
                    break
                    
                if abs(fx_new) < abs(best_fx):
                    best_x, best_fx = x_new, fx_new
                if abs(fx_new) <= 1e-8:
                    return x_new
                    
                x_prev = x_secant
                fx_prev = fx_secant
                x_secant = x_new
                fx_secant = fx_new

    # Return best point found
    return best_x
