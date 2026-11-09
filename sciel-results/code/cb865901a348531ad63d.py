"""Root finding for ill-conditioned equations: Multi-start Global Search with Hybrid Solvers."""
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
    
    try:
        best_f = f(best_x)
        eval_count += 1
    except Exception:
        best_f = float('inf')

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f
        if check_budget():
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            if abs(val) < abs(best_f):
                best_f = val
                best_x = x
            return val
        except Exception:
            return None

    # 1. Initial Check
    if best_f is not None and abs(best_f) <= 1e-8:
        return best_x

    # 2. Multi-start Strategy
    # We will try to solve the equation starting from x0, and if that fails or takes too long,
    # we will expand the search space in a structured way.
    # For ill-conditioned problems, a robust bracketing method (Bisection) is often safer than
    # pure Newton, but slower. We use a hybrid: 
    # - Try to bracket the root.
    # - If bracketed, use Bisection (guaranteed convergence) or Illinois/Regula Falsi (faster).
    # - If not bracketed, use Secant (superlinear) with safeguards.
    
    # Helper: Solve from a given start point with a specific method and budget
    def solve_hybrid(x_start, max_steps, use_bracketing=True):
        nonlocal eval_count, best_x, best_f
        
        x = float(x_start)
        x_prev = None
        f_x = safe_eval(x)
        if f_x is None:
            return None
        
        if abs(f_x) <= 1e-8:
            return x
        
        # If we want bracketing, we need to find a sign change.
        # We'll maintain a bracket [a, b] if possible.
        a, b = None, None
        
        # Initial expansion to find a bracket
        if use_bracketing:
            # Try to find a sign change around x
            step_size = 1.0
            found_bracket = False
            for _ in range(10):
                if check_budget():
                    break
                # Check x + step
                x_plus = x + step_size
                f_plus = safe_eval(x_plus)
                if f_plus is not None:
                    if f_x * f_plus < 0:
                        a, b = x, x_plus
                        found_bracket = True
                        break
                    # Update best if closer
                    if abs(f_plus) < abs(f_x):
                        x = x_plus
                        f_x = f_plus
                        # Reset expansion center? No, keep expanding from original x for bracket logic
                        # But for Newton/Secant, we want to move towards the root.
                        # Let's just update x for the non-bracketing path.
                
                # Check x - step
                x_minus = x - step_size
                f_minus = safe_eval(x_minus)
                if f_minus is not None:
                    if f_x * f_minus < 0:
                        a, b = x_minus, x
                        found_bracket = True
                        break
                    if abs(f_minus) < abs(f_x):
                        x = x_minus
                        f_x = f_minus
                
                step_size *= 2.0
            
            if found_bracket:
                # We have a bracket [a, b]. Use Bisection/Regula Falsi.
                # Regula Falsi (False Position) is better than bisection for monotonic functions.
                fa = f_x if a == x else safe_eval(a)
                fb = f_x if b == x else safe_eval(b)
                if fa is None or fb is None:
                    return None
                
                for _ in range(max_steps):
                    if check_budget():
                        break
                    # Regula Falsi step
                    if fa == fb:
                        # Degenerate, fallback to bisection
                        c = (a + b) / 2.0
                    else:
                        c = (a * fb - b * fa) / (fb - fa)
                    
                    # Ensure c is within bracket to maintain convergence
                    if c <= a or c >= b:
                        c = (a + b) / 2.0
                    
                    fc = safe_eval(c)
                    if fc is None:
                        break
                    
                    if abs(fc) <= 1e-8:
                        return c
                    
                    if fa * fc < 0:
                        b, fb = c, fc
                    else:
                        a, fa = c, fc
                    
                    # If interval is small enough, we are done
                    if abs(b - a) < 1e-12:
                        return (a + b) / 2.0
            else:
                # No bracket found. Use Secant/Newton hybrid.
                pass
        
        # Non-bracketing or failed bracketing: Secant Method
        # We need two points. We have x. Let's get x_prev.
        # If we didn't find a bracket, we might have moved x.
        # Let's restart secant from the best x found so far in this attempt.
        
        # Actually, let's just use a robust Secant implementation from the current best x.
        x_curr = x
        f_curr = f_x
        
        # Get a second point
        h = 1e-4 * max(1.0, abs(x_curr))
        x_prev = x_curr + h
        f_prev = safe_eval(x_prev)
        if f_prev is None:
            return None
        
        if abs(f_prev) < abs(f_curr):
            x_curr, f_curr = x_prev, f_prev
            x_prev, f_prev = x_curr - h, safe_eval(x_curr - h)
            if f_prev is None:
                return None
        
        for _ in range(max_steps):
            if check_budget():
                break
            
            # Secant step
            if f_curr == f_prev:
                # Fallback to bisection-like step or break
                break
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            # Safeguard: Don't jump too far
            max_jump = max(1.0, 0.5 * abs(x_curr))
            if abs(x_new - x_curr) > max_jump:
                x_new = x_curr + math.copysign(max_jump, x_new - x_curr)
            
            f_new = safe_eval(x_new)
            if f_new is None:
                break
            
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Update points
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            
            # Stagnation check
            if abs(x_new - x_prev) < 1e-14:
                break
        
        return x_curr

    # Main Loop: Try multiple strategies/starting points
    
    # Strategy 1: Direct hybrid from x0
    result = solve_hybrid(x0, max_steps=50, use_bracketing=True)
    if result is not None:
        f_res = safe_eval(result)
        if f_res is not None and abs(f_res) <= 1e-8:
            return result
    
    # Strategy 2: Try perturbed starts if direct failed
    # Ill-conditioned problems might have a root close to x0 but Newton diverges.
    # Small perturbations might help.
    perturbations = [1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1]
    for p in perturbations:
        if check_budget():
            break
        x_pert = x0 + p
        result = solve_hybrid(x_pert, max_steps=30, use_bracketing=True)
        if result is not None:
            f_res = safe_eval(result)
            if f_res is not None and abs(f_res) <= 1e-8:
                return result

    # Strategy 3: Broader search
    # If local searches failed, try expanding the search radius.
    # This is expensive, so we do it sparingly.
    radii = [1.0, 10.0, 100.0]
    for r in radii:
        if check_budget():
            break
        # Sample a few points in [x0-r, x0+r]
        samples = [x0 - r, x0 - r/2, x0 + r/2, x0 + r]
        for x_s in samples:
            if check_budget():
                break
            result = solve_hybrid(x_s, max_steps=20, use_bracketing=True)
            if result is not None:
                f_res = safe_eval(result)
                if f_res is not None and abs(f_res) <= 1e-8:
                    return result

    # Return best found
    return best_x
# EVOLVE-BLOCK-END
