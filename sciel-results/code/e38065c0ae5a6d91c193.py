import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 400
    start_time = time.time()
    DEADLINE = start_time + 110
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = x
            return val
        except Exception:
            return float('inf')

    # Initial evaluation
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    
    a = x0
    b = None
    fa = fx0
    bracket_found = False
    
    # Phase 1: Adaptive frequency-sweep bracketing
    # First, estimate the dominant period by sampling at various frequencies
    # Then use that period to place minimal brackets
    
    # Try a set of candidate periods (corresponding to frequencies k = 0.5 to 8)
    # For each, check if f changes sign over one period starting from x0
    candidate_k_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    best_period = None
    best_sign_changes = 0
    
    for k in candidate_k_values:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        period = 2.0 * math.pi / k
        # Sample 4 points over one period: 0, P/4, P/2, 3P/4
        samples = [x0, x0 + period/4, x0 + period/2, x0 + 3*period/4]
        fvals = []
        valid = True
        for s in samples:
            fv = safe_f(s)
            if math.isinf(fv):
                valid = False
                break
            fvals.append(fv)
        if not valid:
            continue
        # Count sign changes
        sc = 0
        for i in range(len(fvals)-1):
            if fvals[i] * fvals[i+1] < 0:
                sc += 1
        if sc > best_sign_changes:
            best_sign_changes = sc
            best_period = period
    
    # Use estimated period to find a bracket
    if best_period is not None:
        # Place points at multiples of half-period to catch sign changes
        half_p = best_period / 2.0
        for n in range(1, 4):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            for sign in [1, -1]:
                x_new = x0 + sign * n * half_p
                f_new = safe_f(x_new)
                if math.isinf(f_new):
                    continue
                if fa * f_new < 0:
                    a = x0
                    b = x_new
                    bracket_found = True
                    break
            if bracket_found:
                break
    
    # Fallback: geometric scan if period estimation failed
    if not bracket_found:
        offsets = [
            0.01, -0.01,
            0.05, -0.05,
            0.1, -0.1,
            0.5, -0.5,
            1.0, -1.0,
            2.0, -2.0,
            5.0, -5.0,
            10.0, -10.0,
        ]
        
        for d in offsets:
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            x_new = x0 + d
            f_new = safe_f(x_new)
            if math.isinf(f_new):
                continue
            if fa * f_new < 0:
                a = x0
                b = x_new
                bracket_found = True
                break
    
    if bracket_found:
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=25)
            if math.isfinite(root):
                return root
        except Exception:
            pass
        
        # Fallback to bisection if brentq fails
        try:
            lo, hi = min(a, b), max(a, b)
            flo, fhi = safe_f(lo), safe_f(hi)
            if flo * fhi > 0:
                raise ValueError("No sign change")
            for _ in range(35):
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                mid = (lo + hi) / 2
                fmid = safe_f(mid)
                if abs(fmid) <= TOL:
                    return mid
                if flo * fmid < 0:
                    hi = mid
                    fhi = fmid
                else:
                    lo = mid
                    flo = fmid
            return (lo + hi) / 2
        except Exception:
            pass
    
    # If no bracket found, try Newton's method from the best point found
    x = best_x
    prev_x = None
    for _ in range(20):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        f_minus = safe_f(x - h)
        f_plus = safe_f(x + h)
        if math.isinf(f_minus) or math.isinf(f_plus):
            break
        d = (f_plus - f_minus) / (2 * h)
        if abs(d) < 1e-15 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
            
        # Damping if step is too large
        max_step = 5.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
             direction = 1 if x_new > x else -1
             x_new = x + direction * max_step
             
        # If we're oscillating, break
        if prev_x is not None and abs(x_new - x) < 1e-10 * max(1.0, abs(x)):
            break
        prev_x = x
        x = x_new
        
    return best_x
