"""Root finder for 3 fixed oscillatory equations using bracketing + Illinois method."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    MAX_CALLS = 490     # Stay well under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        val = f(x)
        if not math.isfinite(val):
            return float('nan')
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = x
        return val

    def brent_like(fa, a, fb, b, tol=1e-8):
        """Brent-like method with safeguarded interpolation."""
        nonlocal calls
        for _ in range(200):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b
            
            # Try secant step
            if fa != fb:
                secant_x = (a * fb - b * fa) / (fb - fa)
            else:
                secant_x = (a + b) / 2.0
            
            # Safeguard: if secant is outside or too close to endpoint, use bisection
            if not (a < secant_x < b) or not math.isfinite(secant_x):
                secant_x = (a + b) / 2.0
            
            # If secant is very close to an endpoint, use bisection
            if abs(secant_x - a) < 1e-14 * max(1.0, abs(a)) or abs(secant_x - b) < 1e-14 * max(1.0, abs(b)):
                secant_x = (a + b) / 2.0
            
            fs = _f(secant_x)
            if math.isnan(fs):
                break
            if abs(fs) <= tol:
                return secant_x
            
            if fa * fs < 0:
                b = secant_x
                fb = fs
            else:
                a = secant_x
                fa = fs
        # Return the better endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    def polish(x_guess, fa_hint=None, fb_hint=None, a_hint=None, b_hint=None):
        """High-precision polish pass using tighter tolerance."""
        nonlocal calls
        if time.time() - start_time > TIME_LIMIT - 2.0:
            return x_guess
        
        # If we have a bracket, use it with tighter tolerance
        if a_hint is not None and b_hint is not None and fa_hint is not None and fb_hint is not None:
            if fa_hint * fb_hint < 0:
                return brent_like(fa_hint, a_hint, fb_hint, b_hint, tol=1e-12)
        
        # Otherwise, do a local search around x_guess
        x = x_guess
        fx = _f(x)
        if math.isnan(fx):
            return x
        if abs(fx) <= 1e-12:
            return x
        
        # Try to find a small bracket around x
        for scale in [1e-4, 1e-5, 1e-6, 1e-7]:
            h = scale * max(1.0, abs(x))
            for sign in [1, -1]:
                xp = x + sign * h
                fp = _f(xp)
                if math.isnan(fp):
                    continue
                if fx * fp < 0:
                    return brent_like(fx, x, fp, xp, tol=1e-12)
                if abs(fp) <= 1e-12:
                    return xp
            if time.time() - start_time > TIME_LIMIT - 2.0:
                break
        
        return x

    # Phase 1: Find a bracket around x0
    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return polish(x, fx)

    # Try expanding in both directions to find a sign change
    base_step = 0.05 * max(1.0, abs(x))
    
    # Try positive direction
    step = base_step
    for i in range(1, 80):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        x_pos = x + i * step
        f_pos = _f(x_pos)
        if math.isnan(f_pos):
            break
        if abs(f_pos) <= 1e-8:
            return polish(x_pos, f_pos)
        if fx * f_pos < 0:
            # Found bracket [x, x_pos]
            result = brent_like(fx, x, f_pos, x_pos)
            return polish(result, None, None, x, x_pos)
        step *= 1.3  # Expand adaptively (slower growth for finer resolution)

    # Try negative direction
    step = base_step
    for i in range(1, 80):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        x_neg = x - i * step
        f_neg = _f(x_neg)
        if math.isnan(f_neg):
            break
        if abs(f_neg) <= 1e-8:
            return polish(x_neg, f_neg)
        if fx * f_neg < 0:
            # Found bracket [x_neg, x]
            result = brent_like(f_neg, x_neg, fx, x)
            return polish(result, None, None, x_neg, x)
        step *= 1.3

    # Phase 2: If no bracket found, do a finer local scan
    # Try a dense local grid around x0 with multiple scales
    for scale in [0.01, 0.005, 0.002, 0.001]:
        local_step = scale * max(1.0, abs(x))
        n_points = 21
        start_x = x - (n_points - 1) / 2.0 * local_step
        prev_x = start_x
        prev_fx = _f(prev_x)
        if math.isnan(prev_fx):
            continue
        if abs(prev_fx) <= 1e-8:
            return polish(prev_x, prev_fx)
        
        for i in range(1, n_points):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            cur_x = start_x + i * local_step
            cur_fx = _f(cur_x)
            if math.isnan(cur_fx):
                break
            if abs(cur_fx) <= 1e-8:
                return polish(cur_x, cur_fx)
            if not math.isnan(prev_fx) and prev_fx * cur_fx < 0:
                result = brent_like(prev_fx, prev_x, cur_fx, cur_x)
                return polish(result, None, None, prev_x, cur_x)
            prev_x = cur_x
            prev_fx = cur_fx
    
    # Phase 3: Try a wider scan with different step sizes
    for mult in [0.5, 2.0, 0.2, 5.0, 1.0, 0.1]:
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break
        wide_step = mult * max(1.0, abs(x))
        prev_x = x - 5 * wide_step
        prev_fx = _f(prev_x)
        if math.isnan(prev_fx):
            continue
        if abs(prev_fx) <= 1e-8:
            return polish(prev_x, prev_fx)
        for i in range(11):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            cur_x = x - 5 * wide_step + i * wide_step
            cur_fx = _f(cur_x)
            if math.isnan(cur_fx):
                break
            if abs(cur_fx) <= 1e-8:
                return polish(cur_x, cur_fx)
            if not math.isnan(prev_fx) and prev_fx * cur_fx < 0:
                result = brent_like(prev_fx, prev_x, cur_fx, cur_x)
                return polish(result, None, None, prev_x, cur_x)
            prev_x = cur_x
            prev_fx = cur_fx

    # Return best x found, with polish if possible
    if best_abs_f < 1e-8:
        return polish(best_x)
    return best_x

# EVOLVE-BLOCK-END
