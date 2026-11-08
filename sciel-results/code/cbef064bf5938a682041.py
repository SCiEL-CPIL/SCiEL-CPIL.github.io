"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
# EVOLVE-BLOCK-START

def _counted_f(f, counter):
    """Wrap f to count evaluations."""
    def wrapped(x):
        counter[0] += 1
        return f(x)
    return wrapped

def _newton_damped(f, x0, max_iter=100, tol=1e-10):
    """Newton's method with adaptive damping."""
    x = float(x0)
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) <= tol:
            return x, True
        h = 1e-7 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        step = fx / d
        # Damp the step to avoid divergence
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
        x_new = x - step
        if not math.isfinite(x_new):
            break
        # Check if we're making progress
        fx_new = f(x_new)
        if abs(fx_new) < abs(fx) * 0.5 or abs(fx_new) <= tol:
            x = x_new
        else:
            # Try smaller step
            x = x - step * 0.5
            if not math.isfinite(x):
                break
    return x, abs(f(x)) <= tol

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    x0 = float(x0)
    counter = [0]
    cf = _counted_f(f, counter)
    
    # Track best result found
    best_x = x0
    best_fx = cf(x0)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Try Newton's method from x0
    x_newton, success = _newton_damped(cf, x0, max_iter=80, tol=1e-10)
    counter[0] += 0  # No extra evals in the call above beyond what was counted
    
    if success:
        return x_newton
    
    fx_newton = cf(x_newton)
    if abs(fx_newton) < best_abs:
        best_x = x_newton
        best_abs = abs(fx_newton)
    
    if best_abs <= 1e-8:
        return best_x
    
    # If Newton didn't converge well, try Brent's method with bracketing
    # First, try to find a bracket around x0
    bracket_found = False
    a = b = None
    fa = fb = None
    
    # Expand bracket around x0
    delta = 1.0
    for attempt in range(20):
        if time.time() > deadline or counter[0] >= 480:
            break
        left = x0 - delta
        right = x0 + delta
        
        fl = cf(left)
        fr = cf(right)
        
        if abs(fl) < best_abs:
            best_x = left
            best_abs = abs(fl)
        if abs(fr) < best_abs:
            best_x = right
            best_abs = abs(fr)
            
        if best_abs <= 1e-8:
            return best_x
        
        if fl * fr < 0:
            a, b = left, right
            fa, fb = fl, fr
            bracket_found = True
            break
        
        delta *= 2.0
        if delta > 1e6:
            break
    
    if bracket_found:
        # Use Brent's method with remaining budget
        remaining_evals = 480 - counter[0]
        if remaining_evals > 10:
            try:
                from scipy.optimize import brentq
                x_brent = brentq(cf, a, b, maxiter=min(50, remaining_evals // 2), xtol=1e-12, rtol=1e-12)
                fx_brent = cf(x_brent)
                if abs(fx_brent) < best_abs:
                    best_x = x_brent
                    best_abs = abs(fx_brent)
            except Exception:
                pass
    
    if best_abs <= 1e-8:
        return best_x
    
    # If still not good enough, try a few more Newton iterations from best_x
    if time.time() < deadline and counter[0] < 470:
        x_try, success2 = _newton_damped(cf, best_x, max_iter=30, tol=1e-10)
        if success2:
            return x_try
        fx_try = cf(x_try)
        if abs(fx_try) < best_abs:
            best_x = x_try
            best_abs = abs(fx_try)
    
    return best_x

# EVOLVE-BLOCK-END
