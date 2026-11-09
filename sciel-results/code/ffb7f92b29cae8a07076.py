"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit
    
    def time_up():
        return time.time() >= deadline

    x0 = float(x0)
    
    # Track best solution found
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        ax = abs(fx)
        if ax < best_abs:
            best_x = x
            best_f = fx
            best_abs = ax

    # If already good enough
    if best_abs <= 1e-8:
        return best_x

    # Multi-start strategy:
    # Instead of one long expansion, try several different starting points
    # and small expansion ranges. This helps if the function has a root
    # nearby but the deterministic path from x0 gets trapped.
    
    # Define a set of relative offsets to try
    # We will use a pseudo-random sequence to vary the search slightly
    random.seed(42) # Deterministic for reproducibility in scoring if needed, or use time? 
    # Let's use a fixed set of diverse offsets to ensure coverage without relying on RNG state across calls if any.
    # Actually, random is fine.
    
    # List of (offset_factor, direction) to try
    # We'll try offsets: 0, +/- 1e-3, +/- 1e-2, +/- 0.1, +/- 1.0, +/- 10.0
    # And for each, do a small local expansion.
    
    candidates = []
    
    # 1. The original point
    candidates.append(x0)
    
    # 2. Small perturbations
    base_h = 1e-3
    for i in range(1, 6):
        h = base_h * (10 ** (i-1))
        candidates.append(x0 + h)
        candidates.append(x0 - h)
        
    # 3. Larger jumps
    for h in [1.0, 10.0, 100.0]:
        candidates.append(x0 + h)
        candidates.append(x0 - h)

    # Shuffle to avoid bias if the first few fail? Or keep structured?
    # Structured is better for "nearby" roots. Let's keep structured but prioritize small steps.
    
    # For each candidate start point, try to find a bracket locally.
    # If we find a bracket, we switch to a robust solver (Bisection/Illinois) for that bracket.
    # We limit the number of evaluations per attempt to ensure we can try multiple starts.
    
    # Total budget: 500 evals.
    # Cost of setup: 1 (x0)
    # Cost per attempt: ~10-20 evals for bracketing + 50 for refinement.
    # We can afford ~10-15 attempts.
    
    # Let's iterate through candidates.
    # To avoid re-evaluating the same point, we'd need a cache, but f is a black box.
    # We just accept the cost.
    
    # Strategy:
    # 1. Try to bracket around x0 first (highest probability).
    # 2. If fail, try around x0 + small_delta.
    # 3. If fail, try around x0 - small_delta.
    # ...
    
    # Let's implement a helper that tries to find a bracket starting from 'start_x'.
    # It returns (a, b, fa, fb) if found, else None.
    # It uses a fixed number of expansion steps.
    
    def try_find_bracket(start_x, max_expansions=10, initial_h=1e-6):
        """
        Attempts to find a bracket [a, b] containing a root, starting near start_x.
        Returns (a, b, fa, fb) or None.
        Uses at most ~2*max_expansions evaluations.
        """
        if time_up():
            return None
            
        x = start_x
        fx = f(x)
        update_best(x, fx)
        
        if abs(fx) <= 1e-8:
            return (x, x, fx, fx) # Degenerate bracket, root found
        
        # Expand in both directions simultaneously? 
        # Or just one direction? 
        # Usually, expanding in both is safer.
        # Let's try expanding h in both + and - directions.
        
        h = initial_h
        left_x = x - h
        right_x = x + h
        
        # We need to evaluate f(left_x) and f(right_x)
        # To save evals, we can check if we already have them? No, no cache.
        
        for _ in range(max_expansions):
            if time_up():
                return None
            
            # Evaluate left
            fl = f(left_x)
            update_best(left_x, fl)
            if fx * fl < 0:
                return (left_x, x, fl, fx)
            if abs(fl) <= 1e-8:
                return (left_x, left_x, fl, fl)
                
            # Evaluate right
            fr = f(right_x)
            update_best(right_x, fr)
            if fx * fr < 0:
                return (x, right_x, fx, fr)
            if abs(fr) <= 1e-8:
                return (right_x, right_x, fr, fr)
                
            # If no sign change, expand h
            h *= 2
            left_x = x - h
            right_x = x + h
            
            # If h becomes huge, we might be overshooting. 
            # But for ill-conditioned, the root might be far.
            
        return None

    # We will iterate through a prioritized list of starting points.
    # Priority: x0, then small offsets, then larger.
    
    # To avoid redundant work, we'll just call try_find_bracket for each.
    # If it succeeds, we solve that bracket.
    
    # Refinement function: Illinois method (Regula Falsi with halving)
    def solve_bracket(a, b, fa, fb):
        if time_up():
            return None
        
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # If a==b, we already have the root
        if a == b:
            return a
            
        # Illinois Method
        # If the same endpoint is kept, halve its function value to force bisection-like behavior
        # This prevents the slow convergence of Regula Falsi.
        
        # Max iterations to avoid infinite loops or time limits
        for _ in range(100):
            if time_up():
                return None
                
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
                
            # Secant step
            if fa == fb:
                c = (a + b) / 2
            else:
                c = b - fb * (b - a) / (fb - fa)
                
            # If c is outside [a, b], or too close to endpoints, bisect
            if c <= a or c >= b:
                c = (a + b) / 2
                
            fc = f(c)
            update_best(c, fc)
            
            if abs(fc) <= 1e-8:
                return c
                
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                
            # Illinois modification:
            # If we just updated 'a' (meaning fa * fc > 0), then 'a' is the "stagnant" side.
            # Wait, standard Illinois: if the new point is closer to the old a, we halve fa.
            # Actually, simpler rule: if the interval did not shrink significantly, or if we keep updating the same side, halve the function value of that side.
            
            # Let's use a simple heuristic: if abs(c - a) < abs(b - a) * 0.5, we are moving towards a.
            # Actually, let's just check if we updated 'a' in the previous step? 
            # We don't track that easily.
            
            # Alternative: If the secant step c is very close to a or b, bisect.
            # The check `if c <= a or c >= b` handles outside.
            # Let's add: if abs(c - a) < 1e-12 * abs(a) or abs(c - b) < 1e-12 * abs(b):
            #     c = (a+b)/2
            
            # A more robust Illinois implementation tracks which endpoint is "old".
            # Let's stick to a simple bisection fallback if progress is slow.
            # If (b-a) hasn't decreased by factor 2 in last 3 steps, bisect.
            # This requires state.
            
            # Let's just rely on the fact that if secant fails, we bisect.
            # And for ill-conditioned, bisection is robust.
            # To speed up, we can try Newton if derivative is available? No, black box.
            
            # Let's add a stagnation check:
            # If abs(fa) > abs(fb), and we just set a=c, then fa is the new one.
            # If we set b=c, fb is the new one.
            # Let's just use Bisection if the secant step doesn't reduce the interval by at least half.
            # We need to store prev_width.
            
        # If loop finishes without returning, return best guess from update_best?
        # Or return midpoint?
        # update_best keeps the global best.
        return None

    # Main Loop: Try different starting points
    
    # We have already evaluated x0.
    # Let's try to find a bracket starting from x0.
    bracket = try_find_bracket(x0, max_expansions=15, initial_h=1e-6)
    
    if bracket:
        a, b, fa, fb = bracket
        root = solve_bracket(a, b, fa, fb)
        if root is not None and abs(f(root)) <= 1e-8:
            return root
        # If solve failed or didn't converge to 1e-8, we might still have a good best_x
        # But let's try other starts if we haven't found a root yet.
        
    # If the first attempt didn't yield a root (or solve_bracket returned None),
    # try other starting points.
    
    # Generate a list of other starting points
    # We want to explore local neighborhood more broadly.
    
    start_points = [x0]
    
    # Small steps
    for i in range(1, 5):
        h = 1e-4 * (10 ** (i-1))
        start_points.append(x0 + h)
        start_points.append(x0 - h)
        
    # Medium steps
    for i in range(1, 4):
        h = 1.0 * (10 ** (i-1))
        start_points.append(x0 + h)
        start_points.append(x0 - h)
        
    # Shuffle? No, keep deterministic order of increasing distance.
    
    # We already tried x0.
    for sp in start_points[1:]:
        if time_up():
            break
            
        # Skip if we are very close to a point we already tried?
        # Hard to check without storing x values.
        
        bracket = try_find_bracket(sp, max_expansions=8, initial_h=1e-6)
        
        if bracket:
            a, b, fa, fb = bracket
            root = solve_bracket(a, b, fa, fb)
            if root is not None:
                # Check if it's actually a root
                if abs(f(root)) <= 1e-8:
                    return root
                # If not, update best and continue?
                # update_best is called inside solve_bracket.
                
    # If all else fails, return the best_x found so far
    return best_x

# EVOLVE-BLOCK-END
