"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses a combination of:
    1. Illinois algorithm (sign-damped secant) for guaranteed convergence
    2. Newton's method with adaptive damping
    3. Bisection for guaranteed convergence
    4. Multiple restarts if initial guess is poor
    5. Deadline guard to stay within time limits
    """
    x0 = float(x0)
    deadline = time.monotonic() + 145.0  # safety margin before 150s kill
    max_evals = 495
    eval_count = [0]
    
    best_x = x0
    best_fx = None
    best_abs = math.inf

    def eval_f(x):
        nonlocal best_x, best_fx, best_abs
        if time.monotonic() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            val = float(val)
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            eval_count[0] += 1
            return None
        eval_count[0] += 1
        if not math.isfinite(val):
            return None
        if abs(val) < best_abs:
            best_x = x
            best_fx = val
            best_abs = abs(val)
        return val

    def check_time():
        return time.monotonic() > deadline

    x = x0
    fx = eval_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Illinois algorithm: sign-damped secant with guaranteed convergence
    def try_illinois(x1, f1, x2, f2, max_iter=80):
        """
        Illinois algorithm: secant method with sign-damped halving.
        Prevents oscillation in ill-conditioned functions.
        """
        if check_time():
            return None
        
        # Ensure we have two distinct points
        if x1 == x2:
            h = 1e-6 * max(1.0, abs(x1))
            x2 = x1 + h
            f2 = eval_f(x2)
            if f2 is None:
                return None
        
        # Track the best point
        bx, bf = x1, f1
        if abs(f2) < abs(f1):
            bx, bf = x2, f2
        
        for i in range(max_iter):
            if check_time():
                break
            
            if abs(f2) <= 1e-12:
                return x2
            if abs(f1) <= 1e-12:
                return x1
            
            # Secant step
            if f1 == f2:
                # Degenerate case: use bisection
                xm = (x1 + x2) / 2
                fm = eval_f(xm)
                if fm is None:
                    break
                if abs(fm) <= 1e-12:
                    return xm
                if abs(fm) < abs(bf):
                    bx, bf = xm, fm
                # Replace the point with larger |f|
                if abs(f1) > abs(f2):
                    x1, f1 = xm, fm
                else:
                    x2, f2 = xm, fm
                continue
            
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x_new):
                break
            
            # Damp if step is too large
            if abs(x_new - x2) > 10 * max(1.0, abs(x2)):
                x_new = x2 + math.copysign(10 * max(1.0, abs(x2)), x_new - x2)
            
            f_new = eval_f(x_new)
            if f_new is None:
                break
            
            if abs(f_new) <= 1e-12:
                return x_new
            
            if abs(f_new) < abs(bf):
                bx, bf = x_new, f_new
            
            # Illinois: if no sign change, halve the larger interval
            if f1 * f_new > 0:
                # No sign change between x1 and x_new; replace x1
                x1, f1 = x2, f2
                # Halve: move x1 halfway toward x2 (the new x1)
                # Actually, standard Illinois: halve the interval that didn't change sign
                # Replace the endpoint with the larger |f|
                if abs(f1) > abs(f2):
                    x1 = (x1 + x2) / 2
                    f1 = eval_f(x1)
                    if f1 is None:
                        break
                    if abs(f1) < abs(bf):
                        bx, bf = x1, f1
                else:
                    x2 = (x1 + x2) / 2
                    f2 = eval_f(x2)
                    if f2 is None:
                        break
                    if abs(f2) < abs(bf):
                        bx, bf = x2, f2
            else:
                # Sign change: standard secant update
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
            
            # Convergence check
            if abs(x2 - x1) < 1e-14 * max(1.0, abs(x2)):
                break
        
        return bx if abs(bf) <= 1e-8 else None

    # Newton's method with damping
    def try_newton(x_start, fx_start, max_iter=30):
        x = x_start
        fx = fx_start
        for i in range(max_iter):
            if check_time():
                return None
            if abs(fx) <= 1e-12:
                return x
            
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d) or abs(d) < 1e-15:
                break
            
            step = fx / d
            # Damp the step if it's too large
            if abs(step) > 10 * max(1.0, abs(x)):
                step = math.copysign(10 * max(1.0, abs(x)), step)
            
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

    # Find a bracket
    def find_bracket(x_center, fx_center, max_expansions=25):
        """Find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        a, b = x_center, x_center
        fa, fb = fx_center, fx_center
        
        # Expand outward to find sign change
        step_size = max(1.0, abs(x_center)) * 0.1
        for _ in range(max_expansions):
            if check_time():
                return None
            
            # Try expanding left
            a_new = a - step_size
            fa_new = eval_f(a_new)
            if fa_new is not None:
                if fa_new * fb < 0:
                    return (a_new, b, fa_new, fb)
                a, fa = a_new, fa_new
            
            # Try expanding right
            b_new = b + step_size
            fb_new = eval_f(b_new)
            if fb_new is not None:
                if fa * fb_new < 0:
                    return (a, b_new, fa, fb_new)
                b, fb = b_new, fb_new
            
            step_size *= 2
        
        return None

    # Bisection
    def try_bisection(a, b, fa, fb, max_iter=60):
        for _ in range(max_iter):
            if check_time():
                break
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
            if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                break
        mid = (a + b) / 2
        return mid if abs(eval_f(mid) if eval_f(mid) is not None else math.inf) <= 1e-8 else None

    # Strategy 1: Illinois from x0 with a nearby point
    h = 1e-5 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        result = try_illinois(x, fx, x2, fx2)
        if result is not None:
            return result

    # Strategy 2: Newton from x0
    result = try_newton(x, fx)
    if result is not None:
        return result

    # Strategy 3: Illinois from a slightly perturbed start
    x_pert = x0 + 1e-3 * max(1.0, abs(x0))
    fx_pert = eval_f(x_pert)
    if fx_pert is not None:
        if abs(fx_pert) <= 1e-8:
            return x_pert
        h2 = 1e-5 * max(1.0, abs(x_pert))
        x2_pert = x_pert + h2
        fx2_pert = eval_f(x2_pert)
        if fx2_pert is not None:
            result = try_illinois(x_pert, fx_pert, x2_pert, fx2_pert)
            if result is not None:
                return result
        result = try_newton(x_pert, fx_pert)
        if result is not None:
            return result

    # Strategy 4: Find bracket and use bisection
    bracket = find_bracket(x, fx)
    if bracket is not None:
        a, b, fa, fb = bracket
        result = try_bisection(a, b, fa, fb)
        if result is not None:
            return result

    # Strategy 5: Multiple restarts with increasing radius
    rng = np.random.RandomState(42)
    for attempt in range(15):
        if check_time():
            break
        
        # Random restart around x0 with increasing radius
        radius = 1e-3 * (10 ** (attempt / 3)) * max(1.0, abs(x0))
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) < abs(best_fx):
            best_x, best_fx = x_try, fx_try
        
        if abs(fx_try) <= 1e-8:
            return x_try
        
        # Try Illinois from this point
        h3 = 1e-5 * max(1.0, abs(x_try))
        x3 = x_try + h3
        fx3 = eval_f(x3)
        if fx3 is not None:
            result = try_illinois(x_try, fx_try, x3, fx3, max_iter=30)
            if result is not None:
                return result
        
        # Try Newton from this point
        result = try_newton(x_try, fx_try, max_iter=15)
        if result is not None:
            return result
        
        # Try to find bracket from this point
        bracket = find_bracket(x_try, fx_try, max_expansions=10)
        if bracket is not None:
            a, b, fa, fb = bracket
            result = try_bisection(a, b, fa, fb, max_iter=30)
            if result is not None:
                return result

    # Return best solution found
    return best_x
# EVOLVE-BLOCK-END
