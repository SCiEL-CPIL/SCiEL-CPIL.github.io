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
        # Use a small perturbation
        h = 1e-5 * max(1.0, abs(x))
        x2 = x + h
        fx2 = safe_eval(x2)
        
        if fx2 is None:
            return x
            
        if abs(fx2) < abs(fx):
            x, fx = x2, fx2
            # Need another point
            x2 = x - h
            fx2 = safe_eval(x2)
            if fx2 is None:
                return x
            if abs(fx2) < abs(fx):
                 x, fx = x2, fx2
        
        # Now we have two points: (x, fx) and (x2, fx2)
        # If they are the same, perturb x
        if x == x2:
             x2 = x + 1e-4
             fx2 = safe_eval(x2)
             if fx2 is None:
                 return x

        # Store the two points
        x1, f1 = x, fx
        x0_, f0_ = x2, fx2
        
        # Iterate
        for _ in range(max_steps):
            if check_budget():
                break
                
            # Check success
            if abs(f1) <= 1e-8:
                return x1
            if abs(f0_) <= 1e-8:
                return x0_

            # Method selection:
            # If we have a bracket, use Bisection for robustness.
            # If not, use Secant.
            
            if f1 * f0_ < 0:
                # Bracketed root. Use Bisection.
                x_new = (x0_ + x1) / 2.0

                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                
                # Update bracket
                if f1 * f_new < 0:
                    x0_, f0_ = x1, f1
                    x1, f1 = x_new, f_new
                else:
                    x0_, f0_ = x_new, f_new
                    
                # Convergence check
                if abs(x1 - x0_) < 1e-12 * max(1.0, abs(x1)):
                    break

            else:
                # Unbracketed. Use Secant.
                if f1 == f0_:
                    # Degenerate, use small step
                    x1 = x1 + 1e-4
                    f1 = safe_eval(x1)
                    if f1 is None: break
                    continue
                
                # Secant step
                x_new = x1 - f1 * (x1 - x0_) / (f1 - f0_)
                
                # Safeguard: Don't jump too far
                max_jump = max(1.0, 0.5 * abs(x1))
                if abs(x_new - x1) > max_jump:
                    x_new = x1 + math.copysign(max_jump, x_new - x1)
                
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                
                # Update: Keep the two points with smallest |f|? Or just the last two?
                # Standard Secant uses last two.
                x0_, f0_ = x1, f1
                x1, f1 = x_new, f_new
                
                # Stagnation
                if abs(x1 - x0_) < 1e-14:
                    break
        
        return x1

    # Main Loop
    # We will try a sequence of starting points.
    # 1. x0
    # 2. x0 +/- small perturbations
    # 3. x0 +/- larger perturbations
    # 4. Global sampling
    
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

    # Strategy 3: Broader sampling to find a sign change or better starting point
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

    # Strategy 4: Final polish pass with tighter tolerance
    # If we're close but not quite there, try a high-precision polish
    if best_abs < 1e-4:
        # Try secant polish from the best point
        h = 1e-10 * max(1.0, abs(best_x))
        x1, f1 = best_x, best_f
        x2, f2 = best_x + h, safe_eval(best_x + h)
        if f2 is not None:
            for _ in range(100):  # Increased iterations for polish
                if check_budget():
                    break
                if abs(f1) <= 1e-8:
                    return x1
                if abs(f2) <= 1e-8:
                    return x2
                if f1 == f2:
                    break
                x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                x2, f2 = x1, f1
                x1, f1 = x_new, f_new
                if abs(x1 - x2) < 1e-15:
                    break

    # Strategy 5: If all else fails, return the best x found so far
    return best_x

# EVOLVE-BLOCK-END
