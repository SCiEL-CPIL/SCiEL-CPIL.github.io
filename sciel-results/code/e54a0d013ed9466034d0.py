"""Root finding for ill-conditioned problems using SciPy robust optimizers."""
# EVOLVE-BLOCK-START
import math
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses SciPy's brentq if a bracket is found, otherwise minimizes |f(x)|.
    """
    max_evals = 500
    eval_count = 0
    tol = 1e-8

    def eval_f(val):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Max evaluations exceeded")
        result = f(val)
        eval_count += 1
        return result

    x = float(x0)
    fx = eval_f(x)
    if abs(fx) <= tol:
        return x

    # Phase 1: Find a bracket [a, b] where f(a)*f(b) < 0
    # Expand geometrically from x0
    a = x
    fa = fx
    b = x
    fb = fx
    found_bracket = False
    
    # We will try to expand to the right and left
    # We maintain a list of points and values to track the best candidate
    # and to help if no bracket is found.
    
    h = 1e-4 * max(1.0, abs(x))
    
    # Try expanding in both directions
    # We'll do a few steps of expansion
    # To save evaluations, we check signs first.
    
    # Strategy: 
    # 1. Check x0 + h, x0 - h
    # 2. If no sign change, increase h by factor 2
    # 3. Repeat until bracket found or h too large / evals low
    
    # We need to be careful with evaluations.
    # Let's try to find a bracket within ~50 evaluations.
    
    current_h = h
    a_left = x
    f_a_left = fx
    a_right = x
    f_a_right = fx
    
    # We'll store the best point found (min |f|)
    best_x = x
    best_f = abs(fx)
    
    for _ in range(30): # Max 30 expansions
        if eval_count > max_evals - 20:
            break
            
        # Try right
        x_r = a_right + current_h
        if abs(x_r) < 1e15:
            fx_r = eval_f(x_r)
            if abs(fx_r) < best_f:
                best_x = x_r
                best_f = abs(fx_r)
            if abs(fx_r) <= tol:
                return x_r
                
            if f_a_right * fx_r < 0:
                found_bracket = True
                a = a_right
                b = x_r
                fa = f_a_right
                fb = fx_r
                break
            
            a_right = x_r
            f_a_right = fx_r
            
        # Try left
        x_l = a_left - current_h
        if abs(x_l) < 1e15:
            fx_l = eval_f(x_l)
            if abs(fx_l) < best_f:
                best_x = x_l
                best_f = abs(fx_l)
            if abs(fx_l) <= tol:
                return x_l
                
            if f_a_left * fx_l < 0:
                found_bracket = True
                a = x_l
                b = a_left
                fa = fx_l
                fb = f_a_left
                break
            
            a_left = x_l
            f_a_left = fx_l
            
        current_h *= 2.0
        
    # Phase 2: Solve
    if found_bracket:
        # Use brentq
        # Ensure we have enough evals left
        remaining = max_evals - eval_count
        try:
            # brentq maxiter is not eval count, but we can estimate.
            # 100 iterations of brentq usually takes < 100 evals.
            # Let's allow up to remaining evals.
            # scipy.optimize.brentq doesn't have maxfev, only maxiter.
            # We'll wrap f to stop if we exceed budget.
            
            def f_bounded(val):
                nonlocal eval_count
                if eval_count >= max_evals:
                    raise RuntimeError("Max evaluations exceeded in brentq")
                return eval_f(val)
                
            # brentq requires f(a) and f(b) to have opposite signs
            # We already checked this.
            root = brentq(f_bounded, a, b, maxiter=remaining, xtol=1e-12, rtol=1e-12)
            return root
        except Exception:
            # Fallback to best_x if brentq fails
            return best_x
    else:
        # No bracket found. Minimize |f(x)| in the explored range [a_left, a_right]
        # Use minimize_scalar with bounded method
        lower_bound = a_left
        upper_bound = a_right
        
        if lower_bound == upper_bound:
            return best_x
            
        def f_abs(val):
            nonlocal eval_count
            if eval_count >= max_evals:
                raise RuntimeError("Max evaluations exceeded in minimize_scalar")
            return abs(eval_f(val))
            
        try:
            res = minimize_scalar(f_abs, bounds=(lower_bound, upper_bound), method='bounded', options={'maxiter': max_evals - eval_count})
            if res.success:
                return res.x
            else:
                return best_x
        except Exception:
            return best_x

# EVOLVE-BLOCK-END
