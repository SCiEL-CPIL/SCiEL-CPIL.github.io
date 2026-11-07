import math
import time

def find_root(f, x0):
    """
    Finds a root x such that |f(x)| <= 1e-8 for a given function f.
    
    Strategy:
    1. Multi-Scale Global Scan: Sample the function at a fixed number of points 
       across a wide interval around x0 to detect any sign change.
       This is robust for oscillatory functions where local methods might miss 
       roots due to step size issues.
    2. Bisection: Once a bracket [a, b] with f(a)*f(b) < 0 is found, use bisection.
       Bisection is guaranteed to converge and has a predictable call count, 
       unlike Newton or Secant methods which can diverge or oscillate.
    3. Deadline Guard: Ensures the function returns within the time limit.
    """
    TOL = 1e-8
    MAX_CALLS = 500
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # 5s margin before 120s limit
    
    calls = 0
    best_x = x0
    best_f = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if math.isfinite(val):
                if abs(val) < abs(best_f):
                    best_f = abs(val)
                    best_x = x
                return val
            else:
                return float('inf')
        except Exception:
            return float('inf')

    x0 = float(x0)
    
    # 1. Check starting point
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0

    # 2. Multi-Scale Global Scan
    # We try to find a bracket [a, b] such that f(a) * f(b) < 0.
    # We scan a wide range to catch oscillations.
    
    # Define search ranges. We expand the range if no root is found in the inner range.
    # The ranges are relative to x0 to handle different scales.
    scale = max(1.0, abs(x0))
    
    # List of (half_width, num_points) to try.
    # We want to cover a wide area but not waste calls.
    # 40 points is a reasonable coarse grid.
    search_configs = [
        (10.0 * scale, 40),
        (100.0 * scale, 40),
        (1000.0 * scale, 40),
        (10000.0 * scale, 40)
    ]
    
    bracket_a = None
    bracket_b = None
    fa = None
    fb = None
    
    for half_width, num_points in search_configs:
        if calls > MAX_CALLS - 20 or time.time() > DEADLINE:
            break
            
        # Generate points from x0 - half_width to x0 + half_width
        # We include x0 in the grid if possible, or just scan linearly.
        # Linear scan is safer for oscillatory functions to not skip over rapid oscillations 
        # if the grid is too coarse, but 40 points over 20*scale is step size ~0.5*scale.
        # If oscillations are faster than that, we might miss them. 
        # However, if x0 is close to a root, the first few points will catch it.
        
        step = (2.0 * half_width) / (num_points - 1)
        
        prev_x = x0 - half_width
        prev_f = safe_f(prev_x)
        
        if abs(prev_f) <= TOL:
            return prev_x
        
        for i in range(1, num_points):
            if calls > MAX_CALLS - 10 or time.time() > DEADLINE:
                break
                
            curr_x = x0 - half_width + i * step
            curr_f = safe_f(curr_x)
            
            if abs(curr_f) <= TOL:
                return curr_x
            
            if math.isfinite(prev_f) and math.isfinite(curr_f):
                if prev_f * curr_f < 0:
                    bracket_a = prev_x
                    bracket_b = curr_x
                    fa = prev_f
                    fb = curr_f
                    break
            
            prev_x = curr_x
            prev_f = curr_f
        
        if bracket_a is not None:
            break

    # 3. If no bracket found in global scan, try a local dense scan around x0
    # This handles the case where the root is very close but the global grid stepped over it
    # due to oscillation, or if x0 is in a "quiet" region.
    if bracket_a is None:
        if calls < MAX_CALLS - 10 and time.time() < DEADLINE:
            local_width = 0.1 * scale
            local_points = 20
            step = (2.0 * local_width) / (local_points - 1)
            prev_x = x0 - local_width
            prev_f = safe_f(prev_x)
            
            if abs(prev_f) <= TOL:
                return prev_x
                
            for i in range(1, local_points):
                if calls > MAX_CALLS - 5 or time.time() > DEADLINE:
                    break
                curr_x = x0 - local_width + i * step
                curr_f = safe_f(curr_x)
                
                if abs(curr_f) <= TOL:
                    return curr_x
                    
                if math.isfinite(prev_f) and math.isfinite(curr_f):
                    if prev_f * curr_f < 0:
                        bracket_a = prev_x
                        bracket_b = curr_x
                        fa = prev_f
                        fb = curr_f
                        break
                
                prev_x = curr_x
                prev_f = curr_f

    # 4. Bisection
    if bracket_a is not None:
        a = bracket_a
        b = bracket_b
        fa = safe_f(a) # Re-evaluate to ensure we have current values in scope for logic, though we have them
        fb = safe_f(b)
        
        # Ensure we have valid finite values
        if not (math.isfinite(fa) and math.isfinite(fb)):
            # Fallback to best_x if bisection setup failed
            return best_x
            
        # Bisection loop
        # Max iterations: log2((b-a)/TOL). 
        # (b-a) is at most 20000*scale. TOL is 1e-8. 
        # log2(2e4 * 1e8) ~ log2(2e12) ~ 41. 
        # So 50 iterations is safe.
        
        for _ in range(50):
            if calls > MAX_CALLS - 2 or time.time() > DEADLINE:
                break
                
            mid = 0.5 * (a + b)
            fmid = safe_f(mid)
            
            if abs(fmid) <= TOL:
                return mid
                
            if not math.isfinite(fmid):
                # If mid is non-finite, shrink the bracket towards the finite side
                # This shouldn't happen often if endpoints are finite, but handle gracefully
                if math.isfinite(fa):
                    b = mid
                    fb = fmid
                elif math.isfinite(fb):
                    a = mid
                    fa = fmid
                else:
                    break
            else:
                # Check sign change
                if fa * fmid < 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
                    
    # 5. Fallback
    return best_x
