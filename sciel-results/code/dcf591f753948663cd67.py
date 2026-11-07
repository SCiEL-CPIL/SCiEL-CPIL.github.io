import math
import time

def find_root(f, x0):
    """
    Finds a root x such that |f(x)| <= 1e-8 for a given function f.
    
    Strategy: Multi-Start Adaptive Secant/Newton Hybrid.
    1. Attempt rapid local convergence using a robust Secant method variant.
    2. If the local attempt fails (divergence, stagnation, or non-finite values),
       perform a "jump" to a new starting point using a geometrically increasing 
       step size. This allows the algorithm to explore the function space 
       efficiently without the high overhead of a dense uniform grid.
    3. Use a strict call budget and deadline guard to ensure termination.
    4. Track the best x found (minimum |f(x)|) to return the best guess if 
       no root is found within tolerance.
    """
    TOL = 1e-8
    MAX_CALLS = 500
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # 5s margin before 120s limit
    
    calls = 0
    best_x = float(x0)
    best_f_val = float('inf')
    
    # Cache to avoid redundant evaluations
    cache = {}
    
    def safe_f(x):
        nonlocal calls, best_x, best_f_val
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
            
        # Check cache
        if x in cache:
            return cache[x]
            
        try:
            val = f(x)
            if not math.isfinite(val):
                val = float('inf')
        except Exception:
            val = float('inf')
            
        calls += 1
        cache[x] = val
        
        # Update best
        if abs(val) < abs(best_f_val):
            best_f_val = abs(val)
            best_x = x
            
        return val

    # Initial check
    fx0 = safe_f(float(x0))
    if abs(fx0) <= TOL:
        return best_x

    # State for the current local search attempt
    x_curr = float(x0)
    f_curr = fx0
    
    # If x0 is bad, we need a second point to start Secant.
    # We use a small initial step.
    h_init = 1e-4 * max(1.0, abs(x0))
    x_prev = x_curr - h_init
    f_prev = safe_f(x_prev)
    
    if abs(f_prev) <= TOL:
        return x_prev

    # We will perform a series of "attempts".
    # Each attempt starts from a specific point and tries to converge.
    # If it fails, we jump to a new point.
    
    # Jump sequence: geometric expansion from x0
    # This is a structural change: instead of a grid, we probe specific points
    # that are increasingly far away, allowing Secant to take over locally.
    
    # Define jump distances
    # We start near x0, then jump further.
    # The jumps are: 1, 10, 100, 1000, ... times base_scale
    # base_scale is max(1, |x0|) to handle scaling.
    
    base_scale = max(1.0, abs(x0))
    jump_factors = [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0]
    
    # We also want to try "local" secant from x0 first.
    # The first "jump" is effectively 0 (start at x0).
    
    # To make the Secant method robust, we implement a version that:
    # 1. Checks for sign change (bracketing) at each step. If found, switch to Bisection for that bracket.
    # 2. If no sign change, continues Secant.
    # 3. If Secant diverges (|x| grows too fast or f grows too fast), breaks.
    
    for jump_idx, factor in enumerate(jump_factors):
        if calls > MAX_CALLS - 20 or time.time() > DEADLINE - 5.0:
            break
            
        # Determine starting point for this attempt
        if jump_idx == 0:
            start_x = x0
            # We already have f(x0) and f(x0 - h_init)
            # Let's restart the secant state cleanly for this "attempt"
            # Actually, we can just continue from the previous state if it was close,
            # but for a "multi-start" strategy, it's cleaner to treat each jump as a new local search.
            # However, re-evaluating is wasteful. 
            # Let's define the start points explicitly.
            pass
        
        # Define the starting point for this jump
        # For jump 0, we are at x0.
        # For jump i > 0, we jump to x0 + factor * base_scale (and maybe x0 - factor * base_scale)
        
        # To be efficient, we'll define a list of candidate start points for each jump level.
        # Level 0: x0
        # Level 1: x0 +/- 5*scale
        # Level 2: x0 +/- 20*scale
        # ...
        
        if jump_idx == 0:
            candidates = [x0]
        else:
            dist = factor * base_scale
            candidates = [x0 + dist, x0 - dist]
            
        for c_x in candidates:
            if calls > MAX_CALLS - 10 or time.time() > DEADLINE - 2.0:
                break
                
            # Start a local search from c_x
            # We need two points to start Secant.
            # Use c_x and c_x + small_step
            
            # If we already evaluated c_x, use it.
            f_c = safe_f(c_x)
            if abs(f_c) <= TOL:
                return c_x
                
            # Get a second point
            # Step size for secant initialization
            step = 1e-3 * max(1.0, abs(c_x))
            c_x2 = c_x + step
            f_c2 = safe_f(c_x2)
            
            if abs(f_c2) <= TOL:
                return c_x2
            
            # Run Secant/Bisection hybrid from (c_x, f_c) and (c_x2, f_c2)
            x1, f1 = c_x, f_c
            x2, f_c2_val = c_x2, f_c2
            
            # Local loop
            local_iter = 0
            max_local_iter = 30 # Limit local iterations to prevent getting stuck
            
            while local_iter < max_local_iter:
                if calls > MAX_CALLS - 5 or time.time() > DEADLINE - 1.0:
                    break
                
                # Check for sign change (bracketing)
                if f1 * f2 < 0:
                    # Switch to Bisection for this bracket
                    a, b = (x1, x2) if f1 < 0 else (x2, x1)
                    fa, fb = (f1, f2) if f1 < 0 else (f2, f1)
                    
                    # Bisection
                    bisect_iter = 0
                    while bisect_iter < 40:
                        if calls > MAX_CALLS - 2 or time.time() > DEADLINE - 0.5:
                            break
                        mid = (a + b) / 2.0
                        f_mid = safe_f(mid)
                        
                        if abs(f_mid) <= TOL:
                            return mid
                            
                        if not math.isfinite(f_mid):
                            break
                            
                        if fa * f_mid <= 0:
                            b = mid
                            fb = f_mid
                        else:
                            a = mid
                            fa = f_mid
                            
                        if abs(b - a) < 1e-14:
                            break
                        bisect_iter += 1
                    break # Exit local loop, try next jump/candidate
                
                # Secant step
                if f1 == f2:
                    break # Degenerate case
                    
                x_next = x2 - f2 * (x2 - x1) / (f2 - f1)
                
                # Damping/Check for divergence
                # If x_next is too far from x2, or f_next is too large, consider it a failure
                # But first, evaluate it.
                f_next = safe_f(x_next)
                
                if abs(f_next) <= TOL:
                    return x_next
                
                # Check for divergence
                # If the step size is increasing rapidly or values are exploding
                if not math.isfinite(f_next) or abs(f_next) > 1e6:
                    break
                    
                # Update points
                x1, f1 = x2, f2
                x2, f2 = x_next, f_next
                
                local_iter += 1
                
            # If we didn't find a root in this local attempt, try next candidate/jump
            # The "best_x" is already updated by safe_f

    return best_x
