"""Root finding for ill-conditioned equations: MODEL-AND-OPTIMIZE with adaptive refinement."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 480   # Safety margin below 500 limit

    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    # Store all evaluated points for potential re-use or as candidates
    eval_history = []  # list of (x, fval)

    try:
        best_f = f(best_x)
        eval_count += 1
        if math.isfinite(best_f):
            best_abs = abs(best_f)
            eval_history.append((best_x, best_f))
    except Exception:
        best_f = float('inf')
        best_abs = float('inf')

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            abs_val = abs(val)
            if abs_val < best_abs:
                best_f = val
                best_x = x
                best_abs = abs_val
            eval_history.append((x, val))
            return val
        except Exception:
            return None

    # 1. Initial Check
    if best_f is not None and best_abs <= 1e-8:
        return best_x

    # Helper: Find a root starting from x_start using a hybrid method
    # Returns the best x found in this local search
    def solve_local(x_start, max_steps=20):
        nonlocal eval_count, best_x, best_f, best_abs
        
        x = float(x_start)
        fx = safe_eval(x)
        if fx is None:
            return None
            
        if abs(fx) <= 1e-8:
            return x

        # Initial step: try to establish a derivative or a second point
        h = 1e-5 * max(1.0, abs(x))
        x2 = x + h
        fx2 = safe_eval(x2)
        
        if fx2 is None:
            return x
            
        if abs(fx2) < abs(fx):
            x, fx = x2, fx2
            x2 = x - h
            fx2 = safe_eval(x2)
            if fx2 is None:
                return x
            if abs(fx2) < abs(fx):
                 x, fx = x2, fx2
        
        if x == x2:
             x2 = x + 1e-4
             fx2 = safe_eval(x2)
             if fx2 is None:
                 return x

        x1, f1 = x, fx
        x0_, f0_ = x2, fx2
        
        for _ in range(max_steps):
            if check_budget():
                break
                
            if abs(f1) <= 1e-8:
                return x1
            if abs(f0_) <= 1e-8:
                return x0_

            if f1 * f0_ < 0:
                # Bracketed root. Use Illinois algorithm for robustness and speed.
                # Illinois is a modified Regula Falsi that prevents stagnation by
                # halving the function value of the retained endpoint.
                
                # Ensure x1 is the "active" end (smaller |f|)
                if abs(f1) > abs(f0_):
                    x1, f1, x0_, f0_ = x0_, f0_, x1, f1

                # Standard Regula Falsi step
                if f1 == f0_:
                    x_new = (x1 + x0_) / 2.0
                else:
                    x_new = (x1 * f0_ - x0_ * f1) / (f0_ - f1)
                
                # Ensure x_new is within the bracket
                if x_new <= min(x1, x0_) or x_new >= max(x1, x0_):
                    x_new = (x1 + x0_) / 2.0

                f_new = safe_eval(x_new)
                if f_new is None:
                    break

                if abs(f_new) <= 1e-8:
                    return x_new

                # Update bracket
                if f1 * f_new < 0:
                    # Root is between x1 and x_new
                    # x1 is the better endpoint (smaller |f|)
                    # We replace x0_ with x_new
                    x0_, f0_ = x_new, f_new
                else:
                    # Root is between x_new and x0_
                    # We replace x1 with x_new
                    # Illinois modification: halve the new function value to force bisection-like behavior
                    # if this side keeps being chosen
                    x1, f1 = x_new, f_new / 2.0

                # Convergence check
                if abs(x1 - x0_) < 1e-12 * max(1.0, abs(x1)):
                    break

            else:
                # Unbracketed. Use Secant.
                if f1 == f0_:
                    x1 = x1 + 1e-4
                    f1 = safe_eval(x1)
                    if f1 is None: break
                    continue
                
                x_new = x1 - f1 * (x1 - x0_) / (f1 - f0_)
                
                max_jump = max(1.0, 0.5 * abs(x1))
                if abs(x_new - x1) > max_jump:
                    x_new = x1 + math.copysign(max_jump, x_new - x1)
                
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                
                x0_, f0_ = x1, f1
                x1, f1 = x_new, f_new
                
                if abs(x1 - x0_) < 1e-14:
                    break
        
        return x1

    # Strategy 1: Direct local solve from x0
    result = solve_local(x0, max_steps=30)
    if result is not None and best_abs <= 1e-8:
        return best_x

    # Strategy 2: Perturbed starts
    perturbations = [
        1e-2, -1e-2,
        1e-3, -1e-3,
        1e-1, -1e-1,
        0.5, -0.5,
        1.0, -1.0
    ]
    
    for p in perturbations:
        if check_budget():
            break
        x_pert = x0 + p
        result = solve_local(x_pert, max_steps=15)
        if result is not None and best_abs <= 1e-8:
            return best_x

    # Strategy 3: Broader sampling
    radii = [10.0, 100.0, 1000.0]
    for r in radii:
        if check_budget():
            break
        samples = [
            x0 - r,
            x0 - r/2,
            x0 + r/2,
            x0 + r,
            x0 - r*0.1,
            x0 + r*0.1
        ]
        
        for x_s in samples:
            if check_budget():
                break
            fs = safe_eval(x_s)
            if fs is not None:
                if best_abs <= 1e-8:
                    return best_x
                result = solve_local(x_s, max_steps=10)
                if result is not None and best_abs <= 1e-8:
                    return best_x

    return best_x

# EVOLVE-BLOCK-END
