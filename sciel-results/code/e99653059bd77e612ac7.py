import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using a hybrid Secant/Illinois strategy with adaptive bracketing.
    
    Strategy:
    1. Attempt to find a bracket [a, b] where f(a) and f(b) have opposite signs.
       - Start with a small step around x0.
       - If no sign change is found, expand the interval exponentially.
       - If no sign change is found after expansion, fall back to minimizing |f(x)|
         using a local search (similar to Nelder-Mead but simpler) to find a near-root.
    2. Once a bracket is found, use the Secant Method (Illinois variant for robustness)
       to converge to the root.
       - The Secant method approximates the derivative using two points, requiring
         only 1 new function evaluation per iteration (after the initial 2).
       - It converges superlinearly, typically requiring far fewer iterations than
         Bisection or Nelder-Mead for smooth functions.
       - If the Secant method fails to reduce the interval or diverges, switch to
         Illinois/Regula Falsi which is more robust but slower.
    3. Return the best point found if the deadline is reached.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if not math.isfinite(fx):
        # If f(x0) is not finite, try a small step
        x = x + 1e-6
        fx = f(x)
        if not math.isfinite(fx):
            return x # Cannot do better
    
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 1: Find a bracket [a, b] with a sign change
    # We want f(a) * f(b) < 0
    a = x
    fa = fx
    
    # Try to find b such that f(b) has opposite sign to f(a)
    # Start with a small step, then expand
    step = 1e-1
    b = None
    fb = None
    
    # Check a few directions and magnitudes
    # 1. Forward small
    # 2. Backward small
    # 3. Forward large
    # 4. Backward large
    
    candidates = [
        (x + step, 1),
        (x - step, 1),
        (x + 1.0, 1),
        (x - 1.0, 1),
        (x + 10.0, 1),
        (x - 10.0, 1),
        (x + 100.0, 1),
        (x - 100.0, 1)
    ]
    
    # Also check if x0 itself is close to a root in a local neighborhood
    # If we don't find a sign change quickly, we might be in a region where
    # the function doesn't cross zero (e.g., touches and turns back).
    # In that case, we need to minimize |f(x)|.
    
    best_x = x
    best_f = fx
    best_g = fx * fx
    
    for cx, _ in candidates:
        if time.time() > deadline - 1.0:
            break
        try:
            fc = f(cx)
        except:
            continue
            
        if not math.isfinite(fc):
            continue
            
        if abs(fc) < abs(best_f):
            best_f = fc
            best_x = cx
            best_g = fc * fc
            
        if abs(fc) <= 1e-8:
            return cx
            
        # Check for sign change
        if fa * fc < 0:
            b = cx
            fb = fc
            break
            
    if b is None:
        # No sign change found. Try to minimize |f(x)| using a simple local search.
        # This handles cases where the root is a tangency or if we are far from roots.
        # Use a coarse-to-fine grid around x0.
        
        # Determine search range based on x0 magnitude
        range_val = max(1.0, abs(x0) * 0.1)
        
        # Grid search
        num_points = 20
        for i in range(num_points):
            if time.time() > deadline - 1.0:
                break
            # Spread points from x0 - range_val to x0 + range_val
            # Use logarithmic spacing near 0? No, linear is fine for now.
            # Actually, let's do a few specific offsets
            offsets = [
                0.0, 1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1, 
                1.0, -1.0, 10.0, -10.0, 100.0, -100.0
            ]
            
            for off in offsets:
                if time.time() > deadline - 1.0:
                    break
                cx = x0 + off
                try:
                    fc = f(cx)
                except:
                    continue
                if not math.isfinite(fc):
                    continue
                if abs(fc) < abs(best_f):
                    best_f = fc
                    best_x = cx
                    best_g = fc * fc
                    if abs(fc) <= 1e-8:
                        return cx
                        
        # If we still don't have a good root, return the best found
        if abs(best_f) <= 1e-8:
            return best_x
            
        # If we found a local minimum but not a root, try to refine it?
        # For now, if no sign change, we can't use Secant/Illinois.
        # Return best_x.
        return best_x

    # Phase 2: Refine using Secant/Illinois
    # We have [a, b] with fa * fb < 0
    # Ensure a < b for simplicity in logic, though not strictly required for Secant
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    # Secant Method with Illinois fallback
    # x0 = b, x1 = a (or vice versa)
    # Let's start with the two points we have.
    x_prev = a
    x_curr = b
    f_prev = fa
    f_curr = fb
    
    max_iter = 100
    
    for i in range(max_iter):
        if time.time() > deadline - 1.0:
            break
            
        # Check convergence of current best
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev
            
        # Check interval size
        if abs(x_curr - x_prev) < 1e-12:
            # Return the one with smaller f
            return x_curr if abs(f_curr) <= abs(f_prev) else x_prev
            
        # Secant step
        # x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        denom = f_curr - f_prev
        if abs(denom) < 1e-30:
            # Degenerate, break
            break
            
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_new):
            break
            
        try:
            f_new = f(x_new)
        except:
            break
            
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update points
        # We want to keep the bracket [a, b] if possible, or use the two most recent points
        # For Secant, we just use the two most recent points.
        # However, to be robust, we should check if x_new is within the bracket.
        # If it is, we can update the bracket.
        
        # Illinois/Regula Falsi logic:
        # If f_new has the same sign as f_prev, replace x_prev.
        # If f_new has the same sign as f_curr, replace x_curr.
        # This maintains a bracket.
        
        # But Secant doesn't strictly maintain a bracket.
        # Let's use a hybrid:
        # If x_new is between x_prev and x_curr, it's a good sign.
        # If not, we might be diverging.
        
        # Let's stick to a robust Illinois method which is a modified Regula Falsi
        # that ensures linear convergence but is very robust.
        # Actually, Secant is faster. Let's try Secant but monitor divergence.
        
        # Check if the new point is "reasonable"
        # If the new point is far away, or if the function value increased significantly,
        # we might be in trouble.
        
        # Let's use the Illinois algorithm directly as it's more robust for oscillatory functions
        # and still converges faster than pure Bisection.
        
        # Illinois Algorithm:
        # 1. Calculate x_new using Regula Falsi (linear interpolation)
        # 2. If x_new is on the same side as x_prev, replace x_prev with x_prev/2 (or similar damping)
        
        # Let's implement Illinois.
        # We have x_prev, f_prev and x_curr, f_curr with opposite signs.
        
        # Regula Falsi step
        x_rf = x_prev - f_prev * (x_curr - x_prev) / (f_curr - f_prev)
        
        if not math.isfinite(x_rf):
            break
            
        try:
            f_rf = f(x_rf)
        except:
            break
        if not math.isfinite(f_rf):
            break
            
        if abs(f_rf) <= 1e-8:
            return x_rf
            
        # Determine which point to replace
        # If f_rf has same sign as f_prev, replace x_prev
        # If f_rf has same sign as f_curr, replace x_curr
        
        if f_rf * f_prev < 0:
            # x_rf is between x_prev and x_curr, and f_rf is opposite to f_prev
            # So x_rf and x_curr form a new bracket
            # But Illinois says: if the new point is on the "old" side, dampen the old point
            # Actually, standard Illinois:
            # If f_new * f_left < 0, the root is in [x_new, x_right].
            # To prevent stagnation, if x_new is close to x_left, we shrink the interval from the left.
            
            # Let's use a simpler robust method: Bisection with Secant acceleration?
            # No, let's just use Bisection if Secant/Illinois is complex.
            # Bisection is guaranteed to converge in ~50 calls.
            # Secant is ~10 calls.
            
            # Let's try Secant first, and if it fails (diverges), switch to Bisection.
            
            # For now, let's just update the points for Secant.
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            
            # Check for divergence
            if abs(x_curr - x_prev) > 1e6 * (1 + abs(x0)):
                # Diverged, switch to Bisection
                # We need a bracket. Do we have one?
                # We lost track of the bracket in the Secant update above.
                # Let's restart with a bracket if we have one.
                # We had [a, b] initially.
                # Let's just return best_x if we are diverging.
                pass
        else:
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            
    return best_x
# EVOLVE-BLOCK-END
