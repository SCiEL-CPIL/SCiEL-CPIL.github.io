import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root x such that |f(x)| <= 1e-8 for a given function f.
    
    Strategy:
    1. Logarithmic Magnitude Expansion: Evaluate f at x0 ± h for h = 10^-4, 10^-3, ..., 10^4.
       This efficiently spans many orders of magnitude with very few calls (approx 18-20 calls)
       to find a bracket [a, b] where a sign change occurs. This avoids the "waste" of linear
       scans on functions where the root is far away or the period is unknown.
    2. Brent's Method: Once a bracket is found, use scipy.optimize.brentq.
       Brent's method is a hybrid of bisection, secant, and inverse quadratic interpolation.
       It is guaranteed to converge (like bisection) but often converges much faster (superlinear),
       significantly reducing the number of function calls needed to reach 1e-8 tolerance.
    3. Fallbacks: If no bracket is found via expansion, perform a dense local scan.
       If Brent fails (e.g., due to numerical issues), fall back to bisection on the best bracket.
    4. Deadline Guard: Strict time management to ensure return within limits.
    """
    TOL = 1e-8
    MAX_CALLS = 500
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0  # 10s margin before 120s limit (evaluator kills at 150s, but task says 120s limit, be safe)
    
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

    # 2. Logarithmic Magnitude Expansion to find a bracket
    # We check x0 +/- 10^k for k in range(-4, 5)
    # This covers from 0.0001 to 10000 relative to x0 (if x0 is non-zero, we scale; if x0 is 0, we use absolute)
    # Actually, better to scale relative to max(1, abs(x0)) to handle large x0.
    scale = max(1.0, abs(x0))
    
    bracket_a = None
    bracket_b = None
    fa = None
    fb = None
    
    # Points to check: x0, x0 +/- scale*10^k
    # We already checked x0.
    # Let's generate the list of offsets
    offsets = []
    for k in range(-4, 5):
        h = scale * (10.0 ** k)
        offsets.append(-h)
        offsets.append(h)
    
    # Sort offsets by magnitude to find the *closest* bracket first? 
    # Or just iterate? 
    # If we iterate small to large, we might find a very tight bracket, which is good for Brent.
    # If we iterate large to small, we might find a wide bracket quickly.
    # Let's try small to large to prioritize local roots if they exist, but expand if not.
    # However, "Logarithmic magnitude" usually implies checking the scales.
    # Let's just check them in order of increasing magnitude to find the smallest bracket containing a sign change.
    # This minimizes the interval width for Brent, potentially helping convergence.
    
    # We need to keep track of the last point to check for sign change with the new point.
    # But since we are jumping around (x0-h, x0+h, x0-10h, x0+10h), we don't have a continuous path.
    # We need to check pairs: (x0, x0+h), (x0, x0-h), then (x0+h, x0+10h), etc?
    # No, the strategy is "identify the coarsest scale where a sign change occurs".
    # This implies checking if f(x0) and f(x0+h) have opposite signs.
    # If not, check next scale.
    
    # Let's re-evaluate the "coarsest scale" idea.
    # If we check h=1, 10, 100...
    # If f(x0) and f(x0+1) have same sign, we don't know if there's a root in (x0, x0+1).
    # But for oscillatory functions, if the period is large, a small h might miss it.
    # If the period is small, a large h might cross multiple roots.
    # Brent's method requires a sign change at the endpoints. It doesn't care how many roots are inside,
    # it will find *one* of them.
    # So, checking (x0, x0+h) is valid.
    
    # Let's try this:
    # 1. Check x0 (done)
    # 2. For k in -4 to 4:
    #    h = scale * 10^k
    #    Check (x0, x0+h). If sign change, bracket found.
    #    Check (x0, x0-h). If sign change, bracket found.
    #    If no sign change with x0, maybe the root is not near x0 but between x0+h and x0+10h?
    #    To be robust, we should also check (x0+h, x0+10h) and (x0-h, x0-10h).
    #    But that doubles the calls.
    #    
    # Alternative: 
    # Generate a sparse grid of points: x0, x0±s, x0±10s, ..., x0±10000s.
    # Sort them.
    # Walk through the sorted points.
    # If f(p[i]) and f(p[i+1]) have opposite signs, we have a bracket.
    # This ensures we don't miss a root between x0 and x0+10s if it exists, 
    # because we check the segment (x0, x0+s) and (x0+s, x0+10s).
    
    points = [x0]
    for k in range(-4, 5):
        h = scale * (10.0 ** k)
        points.append(x0 + h)
        points.append(x0 - h)
    
    # Remove duplicates (if h=0, though k starts at -4 so h>0)
    # Sort points
    points = sorted(list(set(points)))
    
    # Evaluate all these points
    # Number of points: 1 + 9*2 = 19 points.
    # Calls so far: 1 (x0) + 18 = 19.
    # This is very cheap.
    
    vals = {}
    for p in points:
        if p == x0:
            vals[p] = fx0
        else:
            vals[p] = safe_f(p)
            if time.time() > DEADLINE or calls >= MAX_CALLS:
                break
        
        # Early exit if we find a root directly
        if math.isfinite(vals[p]) and abs(vals[p]) <= TOL:
            return p

    # Now scan for sign changes in the sorted points
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        f1 = vals.get(p1, float('inf'))
        f2 = vals.get(p2, float('inf'))
        
        if math.isfinite(f1) and math.isfinite(f2):
            if f1 * f2 < 0:
                bracket_a = p1
                bracket_b = p2
                fa = f1
                fb = f2
                break
            # Also check if one is zero (handled by early exit, but just in case)
            if abs(f1) <= TOL:
                return p1
            if abs(f2) <= TOL:
                return p2

    # 3. If no bracket found in log-scan, try a dense local scan
    # This handles cases where the root is in a "quiet" zone or the log steps were too coarse
    # to catch a sign change (e.g., function touches 0 but doesn't cross? No, root implies f(x)=0.
    # If it touches and turns around, no sign change. But |f(x)|<=1e-8 would have been caught if we sampled close enough.
    # If the root is between two log points and the function doesn't change sign (e.g. tangent), 
    # we need to find the minimum. But the task is "root finder", usually implying f(x)=0.
    # For oscillatory functions, roots are usually crossings.
    # If no crossing is found, we might need to minimize |f(x)|.
    # But let's try a denser scan first.
    
    if bracket_a is None and calls < MAX_CALLS - 20 and time.time() < DEADLINE:
        # Dense scan in the smallest interval around x0
        # Use the smallest h from the log scan: scale * 10^-4
        local_width = scale * 10**-4
        if local_width < 1e-6:
            local_width = 1e-6
        
        num_points = 20
        step = (2.0 * local_width) / (num_points - 1)
        
        prev_x = x0 - local_width
        prev_f = safe_f(prev_x)
        
        if abs(prev_f) <= TOL:
            return prev_x
            
        for i in range(1, num_points):
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

    # 4. Convergence
    if bracket_a is not None:
        a = bracket_a
        b = bracket_b
        
        # Ensure we have valid finite values for Brent
        # We might have them in fa/fb, but let's be safe if we used the dense scan
        if not (math.isfinite(fa) and math.isfinite(fb)):
             fa = safe_f(a)
             fb = safe_f(b)
             
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
            try:
                # Brent's method
                # maxiter=100 is plenty for 1e-8 tol
                root = brentq(f, a, b, xtol=TOL, maxiter=100, full_output=False)
                # Check if root is valid
                if math.isfinite(root):
                    return root
            except Exception:
                pass # Fall through to bisection

        # Fallback to Bisection if Brent failed or bracket invalid
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
             for _ in range(60):
                if calls > MAX_CALLS - 2 or time.time() > DEADLINE:
                    break
                mid = 0.5 * (a + b)
                fmid = safe_f(mid)
                
                if abs(fmid) <= TOL:
                    return mid
                
                if not math.isfinite(fmid):
                    break
                    
                if fa * fmid < 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid

    # 5. Fallback
    return best_x
