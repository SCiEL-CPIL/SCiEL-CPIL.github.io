import math
import time

def find_root(f, x0):
    """
    Finds a root x such that |f(x)| <= 1e-8 for a given function f.
    Strategy:
    1. Global Adaptive Grid Search: Coarse-to-fine uniform sampling to find a sign change bracket.
       This is robust for oscillatory functions where local methods might fail to find a bracket
       if x0 is far from a root or in a region with no sign change nearby.
    2. Bisection Method: Once a bracket [a, b] with f(a)*f(b) <= 0 is found, use bisection
       to converge to the root. Bisection is guaranteed to converge linearly and never diverges,
       unlike secant or Newton methods which can fail on oscillatory or non-smooth functions.
    3. Deadline Guard: Ensures the function returns within the time limit.
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

    # 1. Initial check
    fx0 = safe_f(float(x0))
    if abs(fx0) <= TOL:
        return best_x

    # 2. Global Adaptive Grid Search
    # We define a search space that expands around x0.
    # Oscillatory functions often have roots within a reasonable distance.
    # We use a coarse grid first, then refine if no sign change is found.
    
    # Determine initial search range
    # Heuristic: start with a range that covers several periods if possible.
    # Since we don't know the period, we use a geometric expansion.
    
    # Strategy: 
    # Phase 1: Coarse scan with N points over a range [-R, R] + x0
    # If no sign change, increase R.
    
    # Let's define a set of candidate ranges to search.
    # x0 might be far from 0, so we center the search around x0.
    
    base_range = max(1.0, abs(x0))
    
    # List of (num_points, range_multiplier)
    # We want to balance resolution and coverage.
    # For oscillatory functions, a sign change is likely within a few "units" of the period.
    # Let's try ranges: 1, 5, 20, 100, 1000, 10000 times base_range.
    
    search_configs = [
        (20, 1.0),
        (50, 5.0),
        (100, 20.0),
        (200, 100.0),
        (400, 1000.0), # Might be too many calls if we iterate all, but we break on find
    ]
    
    bracket_a = None
    bracket_b = None
    
    for num_points, mult in search_configs:
        if time.time() > DEADLINE - 5.0:
            break
        if calls > MAX_CALLS - 50: # Reserve calls for bisection
            break
            
        r = base_range * mult
        start = x0 - r
        end = x0 + r
        
        # Generate grid points
        # We include x0 in the grid if possible, but uniform is fine.
        # Step size
        if num_points < 2:
            continue
            
        step = (end - start) / (num_points - 1)
        
        # Evaluate grid
        prev_x = None
        prev_f = None
        
        for i in range(num_points):
            if time.time() > DEADLINE - 2.0:
                break
            if calls > MAX_CALLS - 20:
                break
                
            x = start + i * step
            
            # Skip if we already evaluated this exact point in a previous config? 
            # The cache handles it, but we still count the "check" logic.
            # Actually, safe_f handles the call count.
            
            f_val = safe_f(x)
            
            if not math.isfinite(f_val):
                prev_x = x
                prev_f = f_val
                continue
            
            if abs(f_val) <= TOL:
                return x
                
            if prev_f is not None and math.isfinite(prev_f):
                if prev_f * f_val < 0:
                    bracket_a = prev_x
                    bracket_b = x
                    break
            elif prev_f is None:
                # First point
                pass
            
            prev_x = x
            prev_f = f_val
            
        if bracket_a is not None:
            break

    # 3. Bisection Method
    if bracket_a is not None and bracket_b is not None:
        a = bracket_a
        b = bracket_b
        fa = safe_f(a)
        fb = safe_f(b)
        
        # Ensure we have a valid bracket
        if not (math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0):
            # Try to find a closer bracket by linear interpolation check? 
            # Or just fail to bracket and return best_x.
            # But if we found a sign change, it should be valid unless non-continuous.
            # The problem states "oscillatory equations", implying continuity.
            pass
        else:
            # Bisection
            # Max iterations: log2((b-a)/TOL_x)
            # TOL for x is not specified, only f(x). 
            # But usually, reducing interval width reduces f(x) if f is Lipschitz.
            # We iterate until |f(mid)| <= TOL or interval is tiny.
            
            max_bisect_iter = 50
            for _ in range(max_bisect_iter):
                if time.time() > DEADLINE - 1.0:
                    break
                if calls > MAX_CALLS - 5:
                    break
                    
                mid = (a + b) / 2.0
                f_mid = safe_f(mid)
                
                if abs(f_mid) <= TOL:
                    return mid
                    
                if not math.isfinite(f_mid):
                    break
                    
                # Check sign change
                if fa * f_mid <= 0:
                    b = mid
                    fb = f_mid
                else:
                    a = mid
                    fa = f_mid
                    
                # Stop if interval is very small
                if abs(b - a) < 1e-14:
                    break

    # 4. Fallback: If no bracket found, try a few secant steps from best_x?
    # Or just return best_x.
    # Given the "oscillatory" nature, a global scan should find a bracket.
    # If it didn't, maybe the root is extremely far or the function doesn't cross zero 
    # (e.g., touches zero). But "root finder" implies crossing or touching.
    # If touching, bisection fails. Secant might work better for touching?
    # But bisection is safer for crossing.
    
    # Let's try a quick local secant search from best_x if we haven't used all calls.
    if calls < MAX_CALLS - 10 and time.time() < DEADLINE - 5.0:
        x_curr = best_x
        f_curr = safe_f(x_curr)
        
        # Try a small step
        h = 1e-3 * max(1.0, abs(x_curr))
        x_prev = x_curr - h
        f_prev = safe_f(x_prev)
        
        for _ in range(20):
            if calls > MAX_CALLS - 5 or time.time() > DEADLINE - 2.0:
                break
            if f_prev == f_curr:
                break
            x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            f_next = safe_f(x_next)
            
            if abs(f_next) <= TOL:
                return x_next
                
            # Update
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_next
            f_curr = f_next
            
            # Damping
            if not math.isfinite(f_curr) or abs(f_curr) > 1e6:
                break

    return best_x
