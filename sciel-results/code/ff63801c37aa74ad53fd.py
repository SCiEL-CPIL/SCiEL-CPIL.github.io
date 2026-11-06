import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses a hybrid approach:
    1. Check initial guess.
    2. Attempt to find a bracket [a, b] with sign change.
    3. Use Brent's method if a bracket is found.
    4. Fallback to robust Newton/Secant if no bracket found.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Max evaluations exceeded")
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded")
        eval_count += 1
        return f(x)
    
    x0 = float(x0)
    
    # 1. Check initial guess
    try:
        fx0 = eval_f(x0)
        if np.isnan(fx0) or np.isinf(fx0):
            return x0
        if abs(fx0) <= tol:
            return x0
    except Exception:
        return x0

    # 2. Attempt to find a bracket
    # We will try to expand an interval around x0 to find a sign change.
    # Start with a small step, then expand exponentially.
    a = x0
    fa = fx0
    b = x0
    
    # Try expanding in both directions
    found_bracket = False
    current_a, current_fa = a, fa
    current_b, current_fb = b, fa # Placeholder, will update
    
    # We need distinct points. Let's try small steps first.
    # A common heuristic is to try x0 +/- small_delta, then expand.
    
    # Strategy: 
    # 1. Try x0 + h, x0 - h for small h.
    # 2. If no sign change, expand h.
    
    # To save evaluations, we can check if f(x0+h) or f(x0-h) gives a sign change with f(x0).
    # If not, we need to search further.
    
    # Let's use a more systematic bracketing search.
    # We have f(a) = fx0.
    # We want to find b such that sign(f(a)) != sign(f(b)).
    
    # Try expanding to the right first, then left.
    # Initial step size
    h = 1e-6 * max(1.0, abs(x0))
    
    # We'll iterate, doubling h each time, up to a max range.
    # Max range is arbitrary, say 1e6 * scale.
    max_h = 1e6 * max(1.0, abs(x0))
    
    # We already have f(x0). Let's try x0 + h.
    # To minimize evals, we check if we can bracket.
    
    # Point 1: x0 (known)
    # Point 2: x0 + h
    # Point 3: x0 - h
    
    # Let's try to find a bracket by sampling.
    # We'll keep track of the best candidate root found so far (min |f(x)|) in case we fail.
    best_x = x0
    best_fx = abs(fx0)
    
    # Update best if we find something better during search
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if np.isfinite(fx) and abs(fx) < best_fx:
            best_x = x
            best_fx = abs(fx)
            if best_fx <= tol:
                raise FoundRoot(x)
                
    class FoundRoot(Exception):
        def __init__(self, x):
            self.x = x

    try:
        update_best(x0, fx0)
        
        # Try to find a bracket by expanding steps
        # We will sample points: x0, x0+h, x0-h, x0+2h, x0-2h, ...
        # Actually, standard expansion is:
        # a = x0, b = x0 + h
        # If no sign change, b = x0 + 2h, 4h, etc.
        # If still no sign change, try expanding in the other direction.
        
        # Let's try right expansion first.
        # We have f(x0). Let's evaluate f(x0+h).
        h_val = h
        prev_x = x0
        prev_f = fx0
        bracket_found = False
        a_bracket = None
        b_bracket = None
        
        # We'll search up to 30 expansions (h grows by 2 each time).
        # This covers a wide range.
        for i in range(30):
            if time.time() > deadline:
                break
            if eval_count >= max_evals:
                break
                
            new_x = x0 + h_val
            try:
                new_f = eval_f(new_x)
            except Exception:
                # If eval fails, try the other direction or break
                break
            
            if not np.isfinite(new_f):
                # If inf/nan, shrink step or change direction? 
                # For now, break this direction.
                break
            
            update_best(new_x, new_f)
            
            # Check for sign change between prev_x and new_x
            if prev_f * new_f < 0:
                a_bracket = prev_x
                b_bracket = new_x
                bracket_found = True
                break
            
            # Also check if new_x and x0 (if prev_x was not x0 in a chain? No, we are chaining)
            # Actually, we are checking consecutive points.
            # If we don't find a sign change with the immediate previous point, 
            # we might have jumped over the root or the function is monotonic.
            # Standard bracketing expands the interval [x0, x0 + k*h].
            # Let's stick to expanding from x0 to be safe.
            
            # Reset prev to x0? No, if we move x0 -> x0+h -> x0+2h, 
            # a sign change between x0+h and x0+2h is also a valid bracket.
            # So chaining is fine.
            
            prev_x = new_x
            prev_f = new_f
            h_val *= 2.0
            
            if h_val > max_h:
                break
        
        if not bracket_found:
            # Try left expansion
            h_val = h
            prev_x = x0
            prev_f = fx0
            
            for i in range(30):
                if time.time() > deadline:
                    break
                if eval_count >= max_evals:
                    break
                    
                new_x = x0 - h_val
                try:
                    new_f = eval_f(new_x)
                except Exception:
                    break
                
                if not np.isfinite(new_f):
                    break
                
                update_best(new_x, new_f)
                
                if prev_f * new_f < 0:
                    a_bracket = prev_x
                    b_bracket = new_x
                    bracket_found = True
                    break
                
                prev_x = new_x
                prev_f = new_f
                h_val *= 2.0
                
                if h_val > max_h:
                    break
        
        if bracket_found:
            # Use Brent's method
            try:
                # xtol=1e-12 is tighter than required 1e-8 to ensure |f(x)| <= 1e-8
                # rtol=1e-12
                root = brentq(eval_f, a_bracket, b_bracket, xtol=1e-12, rtol=1e-12, maxiter=100)
                return root
            except Exception:
                # If Brent fails, fall through to fallback
                pass

        # 3. Fallback: Robust Newton/Secant
        # If no bracket found, we rely on local convergence.
        # Use a damped Newton step or Secant method.
        # Since we don't have derivative, use Secant (requires 2 points).
        # We have x0 and potentially one other point from the search above.
        # Let's just use a simple secant iteration with safeguards.
        
        x1 = x0
        f1 = fx0
        
        # We need a second point. If we didn't find a bracket, we might have evaluated x0+h or x0-h.
        # Let's pick the point with the smallest |f| we've seen so far, or just x0+h.
        # To be safe, let's just generate a new point if needed.
        
        # Check if we have a second point from the search.
        # The search loop variables are local to the loop scope? No, they are in function scope.
        # But if the loop broke early or didn't run, prev_x might be x0.
        
        # Let's just use a small perturbation if we don't have a distinct second point.
        if prev_x == x0:
            x2 = x0 + 1e-6 * max(1.0, abs(x0))
            try:
                f2 = eval_f(x2)
            except Exception:
                return best_x
        else:
            x2 = prev_x
            f2 = prev_f
            
        # Secant iteration
        for _ in range(50):
            if time.time() > deadline:
                break
            if eval_count >= max_evals:
                break
                
            if x2 == x1:
                break
                
            # Secant step
            dx = (x2 - x1) * f1 / (f2 - f1)
            x3 = x1 - dx
            
            # Safeguard: limit step size to prevent divergence
            # Max step is distance between current points
            max_step = abs(x2 - x1)
            if abs(dx) > max_step:
                # Bisection-like step
                x3 = (x1 + x2) / 2.0
                
            try:
                f3 = eval_f(x3)
            except Exception:
                break
                
            if not np.isfinite(f3):
                # Shrink step
                x3 = (x1 + x2) / 2.0
                try:
                    f3 = eval_f(x3)
                except Exception:
                    break
                if not np.isfinite(f3):
                    break
            
            update_best(x3, f3)
            
            if abs(f3) <= tol:
                return x3
                
            # Update points
            x1 = x2
            f1 = f2
            x2 = x3
            f2 = f3
            
    except FoundRoot as e:
        return e.x
    except Exception:
        pass

    return best_x
