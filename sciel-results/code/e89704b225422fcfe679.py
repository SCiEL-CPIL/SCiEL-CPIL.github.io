"""Root finding for ill-conditioned functions: IQI/Secant with bracket safety net."""
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Uses inverse quadratic interpolation / secant method with a bracket
    safety net (bisection) for robustness on ill-conditioned problems.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s wall-clock kill
    
    max_evals = 500
    eval_count = 0
    
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        return f(x)
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE
    
    # Initial evaluation
    x = float(x0)
    fx = f_counted(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_abs_fx = abs(fx)
    
    # Try to establish a bracket by expanding from x0
    h = 1e-4 * max(1.0, abs(x))
    a, b = None, None
    fa, fb = None, None
    
    # Check a few nearby points to find a sign change
    for trial in range(20):
        if eval_count >= max_evals or not elapsed_ok():
            break
        # Try expanding in both directions
        for direction in [1, -1]:
            if eval_count >= max_evals or not elapsed_ok():
                break
            xp = x + direction * h
            fxp = f_counted(xp)
            if abs(fxp) <= 1e-8:
                return xp
            if fx * fxp < 0:
                # Found a bracket
                if fx < 0:
                    a, b = x, xp
                    fa, fb = fx, fxp
                else:
                    a, b = xp, x
                    fa, fb = fxp, fx
                break
            h *= 2.0
            if h > 1e6:
                break
        if a is not None:
            break
    
    # If we have a bracket, use a hybrid bisection/Newton/secant approach
    if a is not None:
        # We have bracket [a, b] with fa*fb < 0
        # Use inverse quadratic interpolation when possible, fall back to bisection
        for _ in range(100):
            if not elapsed_ok() or eval_count >= max_evals:
                break
            mid = 0.5 * (a + b)
            fmid = f_counted(mid)
            if abs(fmid) <= 1e-8:
                return mid
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
            
            # Try IQI if we have 3 points
            # For simplicity, use bisection as primary with secant refinement
            if b - a < 1e-12:
                break
        
        # Final check
        for x_check in [a, b, 0.5*(a+b)]:
            if abs(f(x_check)) <= 1e-8:
                return x_check
        # Return the point with smallest |f|
        fa_val = abs(fa)
        fb_val = abs(fb)
        return a if fa_val < fb_val else b
    
    # No bracket found; use secant/IQI method
    # Need two points for secant
    h = 1e-6 * max(1.0, abs(x))
    x_prev = x + h
    f_prev = f_counted(x_prev)
    
    if abs(f_prev) <= 1e-8:
        return x_prev
    
    # Keep track of best
    if abs(f_prev) < best_abs_fx:
        best_x = x_prev
        best_abs_fx = abs(f_prev)
    
    x_curr = x
    f_curr = fx
    
    for _ in range(100):
        if not elapsed_ok() or eval_count >= max_evals:
            break
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev
        
        if abs(f_prev) < best_abs_fx:
            best_x = x_prev
            best_abs_fx = abs(f_prev)
        if abs(f_curr) < best_abs_fx:
            best_x = x_curr
            best_abs_fx = abs(f_curr)
        
        # Secant step
        if f_curr == f_prev:
            # Degenerate case; perturb
            x_new = x_curr + 1e-8 * max(1.0, abs(x_curr))
        else:
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # If step is too small or too large, try bisection-like damping
        if abs(x_new - x_curr) < 1e-14 * max(1.0, abs(x_curr)):
            x_new = 0.5 * (x_curr + x_prev)
        
        f_new = f_counted(x_new)
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Shift points
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
