"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses a combination of:
    1. Newton's method with adaptive damping
    2. Secant method fallback
    3. Bisection for guaranteed convergence
    4. Multiple restarts if initial guess is poor
    5. Deadline guard to stay within time limits
    """
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    max_evals = 450
    eval_count = [0]
    
    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            return val
        except (ValueError, OverflowError, ZeroDivisionError):
            eval_count[0] += 1
            return None
    
    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Track best solution found so far
    best_x = x
    best_fx = fx
    
    # Try Newton's method with damping
    def try_newton(x_start, fx_start, max_iter=30):
        x = x_start
        fx = fx_start
        for i in range(max_iter):
            if time.time() > deadline:
                return None
            if abs(fx) <= 1e-12:
                return x
            
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            
            step = fx / d
            # Damp the step if it's too large
            if abs(step) > abs(x) * 0.5:
                step = math.copysign(min(abs(step), abs(x) * 0.5), step)
            
            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try smaller step
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break
        return x if abs(fx) <= 1e-8 else None
    
    # Try secant method
    def try_secant(x1, fx1, x2, fx2, max_iter=30):
        for i in range(max_iter):
            if time.time() > deadline:
                return None
            if abs(fx1) <= 1e-12:
                return x1
            if abs(fx2) <= 1e-12:
                return x2
            
            if fx1 == fx2:
                break
            
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new
        
        best = x1 if abs(fx1) <= abs(fx2) else x2
        best_fx = fx1 if abs(fx1) <= abs(fx2) else fx2
        return best if abs(best_fx) <= 1e-8 else None
    
    # Try bisection - need a bracket
    def find_bracket(x_center, fx_center, max_expansions=20):
        """Find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        a, b = x_center, x_center
        fa, fb = fx_center, fx_center
        
        # Expand outward to find sign change
        step_size = max(1.0, abs(x_center)) * 0.1
        for _ in range(max_expansions):
            if time.time() > deadline:
                return None
            
            # Try expanding left
            a_new = a - step_size
            fa_new = eval_f(a_new)
            if fa_new is not None and fa_new * fb < 0:
                return (a_new, b, fa_new, fb)
            if fa_new is not None:
                a, fa = a_new, fa_new
            
            # Try expanding right
            b_new = b + step_size
            fb_new = eval_f(b_new)
            if fb_new is not None and fa * fb_new < 0:
                return (a, b_new, fa, fb_new)
            if fb_new is not None:
                b, fb = b_new, fb_new
            
            step_size *= 2
        
        return None
    
    def try_bisection(a, b, fa, fb, max_iter=50):
        for _ in range(max_iter):
            if time.time() > deadline:
                return None
            mid = (a + b) / 2
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        mid = (a + b) / 2
        return mid
    
    # First, try Newton from x0
    result = try_newton(x, fx)
    if result is not None:
        return result
    
    # Update best
    if abs(fx) < abs(best_fx):
        best_x, best_fx = x, fx
    
    # Try secant with a perturbed point
    h = 1e-5 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        result = try_secant(x, fx, x2, fx2)
        if result is not None:
            return result
    
    # Try to find a bracket and use bisection
    bracket = find_bracket(x, fx)
    if bracket is not None:
        a, b, fa, fb = bracket
        result = try_bisection(a, b, fa, fb)
        if result is not None:
            return result
    
    # Multiple restarts with random perturbations
    rng = np.random.RandomState(42)
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Random restart around x0 with increasing radius
        radius = 1e-3 * (10 ** attempt) * max(1.0, abs(x0))
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) < abs(best_fx):
            best_x, best_fx = x_try, fx_try
        
        if abs(fx_try) <= 1e-8:
            return x_try
        
        # Try Newton from this point
        result = try_newton(x_try, fx_try, max_iter=15)
        if result is not None:
            return result
    
    # Return best solution found
    return best_x
# EVOLVE-BLOCK-END
