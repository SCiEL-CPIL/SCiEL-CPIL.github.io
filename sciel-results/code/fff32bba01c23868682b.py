"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust combination of:
    1. Inverse quadratic interpolation with adaptive step control
    2. Secant method as fallback
    3. Bisection when bracketing is available
    4. Multiple restart strategies if initial attempts fail
    """
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    eval_count = [0]
    best_x = x0
    best_fx = None
    
    def eval_f(x):
        nonlocal best_x, best_fx
        if time.time() > deadline:
            return None
        fx = f(x)
        eval_count[0] += 1
        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        return fx
    
    def time_exceeded():
        return time.time() > deadline or eval_count[0] >= 500
    
    # First evaluation
    fx0 = eval_f(x0)
    if fx0 is None:
        return best_x
    if abs(fx0) <= 1e-8:
        return x0
    
    # Try to find a sign change for bisection bracket
    # Explore around x0 to find a bracket
    bracket = None
    h = 1.0
    for scale in [1.0, 0.1, 0.01, 10.0, 100.0]:
        if time_exceeded():
            break
        h = scale
        x_left = x0 - h
        x_right = x0 + h
        fx_left = eval_f(x_left)
        if fx_left is None:
            continue
        if abs(fx_left) <= 1e-8:
            return x_left
        fx_right = eval_f(x_right)
        if fx_right is None:
            continue
        if abs(fx_right) <= 1e-8:
            return x_right
        if fx_left * fx_right < 0:
            bracket = (x_left, x_right, fx_left, fx_right)
            break
    
    # Main iteration loop using inverse quadratic interpolation / secant hybrid
    # Maintain history of (x, fx) for interpolation
    history = [(x0, fx0)]
    
    if bracket:
        a, b, fa, fb = bracket
        history.append((a, fa))
        history.append((b, fb))
    
    # If we have a bracket, we can use it for safety
    for iteration in range(100):
        if time_exceeded():
            break
        
        # Get the last two points for secant
        if len(history) < 2:
            # Need at least 2 points; do a small step
            x_new = x0 + 1e-6 * max(1.0, abs(x0))
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            history.append((x_new, fx_new))
            continue
        
        x1, fx1 = history[-1]
        x2, fx2 = history[-2]
        
        if abs(fx1) <= 1e-8:
            return x1
        if abs(fx2) <= 1e-8:
            return x2
        
        # Try inverse quadratic interpolation if we have 3 points
        x_new = None
        if len(history) >= 3:
            x3, fx3 = history[-3]
            # Inverse quadratic interpolation
            try:
                if fx1 != fx2 and fx2 != fx3 and fx1 != fx3:
                    # Formula for inverse quadratic interpolation
                    L1 = fx1 / ((fx1 - fx2) * (fx1 - fx3))
                    L2 = fx2 / ((fx2 - fx1) * (fx2 - fx3))
                    L3 = fx3 / ((fx3 - fx1) * (fx3 - fx2))
                    x_new = L1 * x1 + L2 * x2 + L3 * x3
            except (ZeroDivisionError, OverflowError):
                x_new = None
        
        if x_new is None or not math.isfinite(x_new):
            # Fall back to secant method
            if fx1 != fx2:
                x_new = x1 - fx1 * (x1 - x2) / (fx1 - fx2)
            else:
                # Degenerate case
                x_new = x1 + 1e-6 * max(1.0, abs(x1))
        
        # If we have a bracket, ensure x_new stays within it
        if bracket:
            a, b, fa, fb = bracket
            if fa * fb < 0:
                if x_new < min(a, b) or x_new > max(a, b):
                    # Use bisection instead
                    x_new = (a + b) / 2.0
        
        # Adaptive step control: limit step size to prevent divergence
        step = abs(x_new - x1)
        max_step = 0.5 * max(1.0, abs(x1))
        if step > max_step:
            # Limit the step
            direction = 1 if x_new > x1 else -1
            x_new = x1 + direction * max_step
        
        fx_new = eval_f(x_new)
        if fx_new is None:
            break
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        # Update bracket if we have one
        if bracket:
            a, b, fa, fb = bracket
            if fa * fx_new <= 0:
                b, fb = x_new, fx_new
            elif fb * fx_new <= 0:
                a, fa = x_new, fx_new
            else:
                # No sign change; keep old bracket but it's less useful
                pass
            bracket = (a, b, fa, fb)
        
        # Update history: keep last 3 points
        history.append((x_new, fx_new))
        if len(history) > 3:
            history.pop(0)
    
    # If we haven't found a root, try a few more restarts with different strategies
    for restart in range(5):
        if time_exceeded():
            break
        
        # Try different initial points
        offsets = [1e-4, -1e-4, 1e-3, -1e-3, 1e-2]
        x_try = x0 + offsets[restart % len(offsets)]
        fx_try = eval_f(x_try)
        if fx_try is None:
            break
        if abs(fx_try) <= 1e-8:
            return x_try
        
        # Do a few secant steps from here
        x_prev = x0
        fx_prev = fx0
        x_curr = x_try
        fx_curr = fx_try
        
        for _ in range(20):
            if time_exceeded():
                break
            if abs(fx_curr) <= 1e-8:
                return x_curr
            if fx_curr == fx_prev:
                break
            x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            fx_next = eval_f(x_next)
            if fx_next is None:
                break
            if abs(fx_next) <= 1e-8:
                return x_next
            x_prev, fx_prev = x_curr, fx_curr
            x_curr, fx_curr = x_next, fx_next
    
    return best_x
# EVOLVE-BLOCK-END
