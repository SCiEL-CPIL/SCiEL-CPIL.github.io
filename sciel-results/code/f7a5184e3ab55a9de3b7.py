"""Derivative-free trust-region root finder for ill-conditioned 1D problems."""
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using derivative-free methods.
    Strategy:
      1. Try to find a bracket around x0 with efficient expansion.
      2. If bracket found, use Illinois method (robust bisection+interpolation).
      3. If no bracket, use adaptive local search on |f(x)|.
    
    Budget: at most 500 evaluations of f.
    """
    target_tol = 1e-8
    max_evals = 500
    eval_count = [0]

    def eval_f(x):
        eval_count[0] += 1
        try:
            val = f(x)
            if not (isinstance(val, (int, float)) or (hasattr(val, '__float__'))):
                val = float(val)
            if math.isnan(val) or math.isinf(val):
                return float('inf')
            return val
        except Exception:
            return float('inf')

    x = float(x0)
    f0 = eval_f(x)
    if abs(f0) <= target_tol:
        return x

    # --- Phase 1: Find a bracket ---
    bracket_found = False
    a, b = None, None
    fa, fb = None, None

    # Try expanding bracket with geometric progression
    h = max(1e-4, 1e-4 * abs(x))
    max_h = 1e10
    
    for _ in range(25):
        if eval_count[0] >= max_evals - 30:
            break
        
        x_plus = x + h
        x_minus = x - h
        f_plus = eval_f(x_plus)
        f_minus = eval_f(x_minus)
        
        if abs(f_plus) <= target_tol:
            return x_plus
        if abs(f_minus) <= target_tol:
            return x_minus
        
        # Check sign changes
        if f0 * f_plus < 0:
            a, b = x, x_plus
            fa, fb = f0, f_plus
            bracket_found = True
            break
        if f0 * f_minus < 0:
            a, b = x_minus, x
            fa, fb = f_minus, f0
            bracket_found = True
            break
        if f_plus * f_minus < 0:
            a, b = x_minus, x_plus
            fa, fb = f_minus, f_plus
            bracket_found = True
            break
        
        h *= 2.0
        if h > max_h:
            break

    if bracket_found:
        # --- Phase 2: Illinois method on the bracket ---
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        for _ in range(200):
            if eval_count[0] >= max_evals - 5:
                break
            
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                if abs(fa) <= abs(fb):
                    return a
                else:
                    return b
            
            if abs(fa) <= target_tol:
                return a
            if abs(fb) <= target_tol:
                return b
            
            if abs(fb - fa) < 1e-30:
                x_new = 0.5 * (a + b)
            else:
                x_new = b - fb * (b - a) / (fb - fa)
            
            if x_new <= a:
                x_new = a + 0.5 * (b - a)
            elif x_new >= b:
                x_new = b - 0.5 * (b - a)
            
            f_new = eval_f(x_new)
            
            if abs(f_new) <= target_tol:
                return x_new
            
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new
            
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa

        if abs(fa) <= abs(fb):
            return a
        else:
            return b

    # --- Phase 3: No bracket found. Local search on |f(x)| ---
    best_x = x
    best_f = abs(f0)
    
    step = max(1e-3, 1e-3 * abs(x))
    
    for iteration in range(150):
        if eval_count[0] >= max_evals - 10:
            break
        
        if best_f <= target_tol:
            break
        
        x_trial = best_x + step
        f_trial = abs(eval_f(x_trial))
        
        if f_trial < best_f:
            best_x = x_trial
            best_f = f_trial
            step *= 1.5
            if step > 1e8:
                step = 1e8
        else:
            x_trial = best_x - step
            f_trial = abs(eval_f(x_trial))
            
            if f_trial < best_f:
                best_x = x_trial
                best_f = f_trial
                step *= 1.5
                if step > 1e8:
                    step = 1e8
            else:
                step *= 0.5
                if step < 1e-15:
                    break
    
    if best_f <= target_tol:
        return best_x

    # --- Phase 4: Final secant attempt ---
    if eval_count[0] >= max_evals - 3:
        return best_x
    
    f_best_signed = eval_f(best_x)
    
    if abs(f_best_signed) <= target_tol:
        return best_x
    
    h = max(1e-6, 1e-6 * abs(best_x))
    x2 = best_x + h
    f2 = eval_f(x2)
    
    if abs(f2) <= target_tol:
        return x2
    
    x_a, f_a = best_x, f_best_signed
    x_b, f_b = x2, f2
    
    for _ in range(30):
        if eval_count[0] >= max_evals - 2:
            break
        
        if abs(f_a) <= target_tol:
            return x_a
        if abs(f_b) <= target_tol:
            return x_b
        
        if abs(f_b - f_a) < 1e-30:
            break
        
        x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
        
        if abs(x_new - x_b) > 10 * abs(x_b - x_a):
            x_new = 0.5 * (x_a + x_b)
        
        f_new = eval_f(x_new)
        
        if abs(f_new) <= target_tol:
            return x_new
        
        if abs(f_new) < abs(f_a):
            x_a, f_a = x_new, f_new
        else:
            x_b, f_b = x_new, f_new
    
    if abs(f_a) <= abs(f_b):
        return x_a
    else:
        return x_b

# EVOLVE-BLOCK-END
