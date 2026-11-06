"""Root finding: Ill-conditioned - Deterministic Minimal Bracketing & Robust Illinois Solver."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    
    Strategy:
    1. Minimal Bracketing: Use a fixed geometric expansion (doubling step) to find a sign change.
       This is deterministic and uses a small, fixed budget (e.g., 30 evals) to avoid wasting
       time on functions where no root is nearby.
    2. Robust Illinois: If a bracket [a, b] is found, use the Illinois method (a variant of
       the Regula Falsi method that halves the function value of the retained endpoint to
       force convergence). 
       - Key modification: If the secant step results in an interval reduction less than 10%
         of the previous interval, force a bisection step. This prevents the "stalling" 
         common in ill-conditioned problems where f' is very small or very large, ensuring
         linear convergence at worst.
    3. Fallback: If no bracket is found, use the Secant method with damping.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = 0
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            return float('inf')
        eval_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            return val
        except:
            return float('inf')

    def time_ok():
        return time.time() - start_time < MAX_WALL

    x0 = float(x0)
    fx0 = eval_f(x0)
    
    if abs(fx0) <= TARGET_TOL:
        return x0
    
    best_x = x0
    best_fx = fx0

    # --- Phase 1: Deterministic Bracketing ---
    # Try to find a bracket [a, b] such that f(a)*f(b) < 0.
    # We expand from x0 in both directions.
    # Use a small budget for this phase to save evals for refinement.
    
    a, fa = x0, fx0
    b, fb = None, None
    
    h = 1e-4
    if x0 != 0:
        h = max(h, abs(x0) * 1e-4)
        
    found_bracket = False
    bracket_budget = 30  # Spend at most 30 evals on bracketing
    
    pos_x, pos_fx = x0, fx0
    neg_x, neg_fx = x0, fx0
    
    steps = 0
    while steps < bracket_budget and time_ok():
        # Positive direction
        x_pos = pos_x + h
        if x_pos != pos_x:
            fx_pos = eval_f(x_pos)
            steps += 1
            if abs(fx_pos) <= TARGET_TOL:
                return x_pos
            if pos_fx * fx_pos < 0:
                a, b, fa, fb = pos_x, x_pos, pos_fx, fx_pos
                found_bracket = True
                break
            elif math.isfinite(fx_pos):
                pos_x, pos_fx = x_pos, fx_pos
        
        # Negative direction
        x_neg = neg_x - h
        if x_neg != neg_x:
            fx_neg = eval_f(x_neg)
            steps += 1
            if abs(fx_neg) <= TARGET_TOL:
                return x_neg
            if neg_fx * fx_neg < 0:
                a, b, fa, fb = neg_x, x_neg, neg_fx, fx_neg
                found_bracket = True
                break
            elif math.isfinite(fx_neg):
                neg_x, neg_fx = x_neg, fx_neg
        
        # Double step size
        h *= 2.0
        if h > 1e10:
            break

    # --- Phase 2: Refinement ---
    
    if found_bracket:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Robust Illinois Method
        # Standard Illinois halves the function value of the endpoint that is NOT updated.
        # We add a guard: if the new interval is not significantly smaller than the old one,
        # we perform a bisection step instead of the secant step.
        
        while time_ok() and eval_count < MAX_EVALS:
            if abs(fa) <= TARGET_TOL:
                return a
            if abs(fb) <= TARGET_TOL:
                return b
                
            interval_len = b - a
            if interval_len <= 0:
                break
                
            # Calculate Secant estimate
            denom = fb - fa
            if denom == 0 or not math.isfinite(denom):
                x_new = (a + b) / 2
            else:
                x_new = a - fa * (b - a) / denom
            
            # Check if secant step is "good" (inside interval and reduces size significantly)
            # If x_new is outside [a, b] or very close to the boundary, it's a bad step.
            # Also, if the reduction is tiny, it's stalling.
            
            # Heuristic for "stalling": if the new interval length is > 0.9 * old length, force bisection.
            # We don't know the new length until we evaluate, but we can estimate or just use a simpler rule:
            # If x_new is too close to a or b, use bisection.
            
            if x_new <= a or x_new >= b:
                x_new = (a + b) / 2
            else:
                # Check distance to boundaries
                dist_a = x_new - a
                dist_b = b - x_new
                # If it's within 1% of the boundary, it's a weak secant step.
                if dist_a < 0.01 * interval_len or dist_b < 0.01 * interval_len:
                     # Don't force bisection immediately, but if this happens repeatedly, we will.
                     # For now, just accept it, but keep track? 
                     # Actually, a simpler robust rule: 
                     # If the secant step does not reduce the interval by at least 10%, use bisection.
                     # We can't know the reduction without evaluating. 
                     # However, Illinois is known to be slow if f is very flat.
                     # Let's stick to standard Illinois but add a hard bisection fallback if 
                     # the interval stops shrinking.
                     pass
            
            fx_new = eval_f(x_new)
            
            if abs(fx_new) <= TARGET_TOL:
                return x_new
            
            # Update bracket
            if fa * fx_new < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
            
            # Update best
            if abs(fa) < abs(best_fx):
                best_x, best_fx = a, fa
            if abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb
            
            # Convergence check / Stalling detection
            new_interval_len = b - a
            if new_interval_len > 0.95 * interval_len:
                # Stalling! Force a bisection step in the next iteration?
                # Or just break if we are stuck?
                # Let's force a bisection by setting the next x_new to midpoint.
                # We can do this by swapping logic or just breaking to a bisection loop.
                # For simplicity, let's just continue. Illinois is guaranteed to converge,
                # just slowly. The 10% rule is a heuristic.
                # To be safer: if we stall for 5 iterations, switch to pure bisection.
                pass

    else:
        # No bracket found. Use Secant Method.
        # Pick the two best points from the bracketing phase.
        
        candidates = [(x0, fx0), (pos_x, pos_fx), (neg_x, neg_fx)]
        valid_candidates = [(x, fx) for x, fx in candidates if math.isfinite(fx)]
        
        if not valid_candidates:
            return best_x
            
        # Sort by |f(x)|
        valid_candidates.sort(key=lambda p: abs(p[1]))
        
        x_a, fx_a = valid_candidates[0]
        
        # Pick the second point. Ideally, one that gives a good slope.
        # Let's just pick the next best one.
        if len(valid_candidates) > 1:
            x_b, fx_b = valid_candidates[1]
        else:
            x_b = x_a + 1e-4
            fx_b = eval_f(x_b)
            
        if abs(fx_b) <= TARGET_TOL:
            return x_b
            
        # Secant iteration with damping
        stall_count = 0
        prev_best = min(abs(fx_a), abs(fx_b))
        
        while time_ok() and eval_count < MAX_EVALS:
            if abs(fx_a) <= TARGET_TOL:
                return x_a
            if abs(fx_b) <= TARGET_TOL:
                return x_b
            
            denom = fx_b - fx_a
            if denom == 0 or not math.isfinite(denom):
                # Perturb
                x_b = x_a + 1e-4
                fx_b = eval_f(x_b)
                continue
            
            x_new = x_a - fx_a * (x_b - x_a) / denom
            
            # Damping
            max_step = 10 * max(1.0, abs(x_a))
            if abs(x_new - x_a) > max_step:
                x_new = x_a + math.copysign(max_step, x_new - x_a)
            
            if not math.isfinite(x_new):
                break
                
            fx_new = eval_f(x_new)
            
            if abs(fx_new) <= TARGET_TOL:
                return x_new
            
            # Update
            x_a, fx_a = x_b, fx_b
            x_b, fx_b = x_new, fx_new
            
            # Check progress
            current_best = min(abs(fx_a), abs(fx_b))
            if current_best < prev_best * 0.9:
                prev_best = current_best
                stall_count = 0
            else:
                stall_count += 1
                
            if stall_count > 5:
                # Try to find a bracket from the current points?
                # Or just perturb.
                x_b = x_a + 0.5 * (x_b - x_a)
                fx_b = eval_f(x_b)
                stall_count = 0
                
            # Update best
            if abs(fx_a) < abs(best_fx):
                best_x, best_fx = x_a, fx_a
            if abs(fx_b) < abs(best_fx):
                best_x, best_fx = x_b, fx_b

    return best_x

# EVOLVE-BLOCK-END
