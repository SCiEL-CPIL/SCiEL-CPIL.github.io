import math
import time
import scipy.optimize as opt

def find_root(f, x0):
    START_TIME = time.time()
    TIME_LIMIT = 145.0
    MAX_CALLS = 495
    calls = 0
    cache = {}
    best_x = x0
    best_f = float('inf')
    found = False

    def safe_f(x):
        nonlocal calls, best_x, best_f, found
        if found:
            return 0.0
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - START_TIME > TIME_LIMIT:
            return float('nan')
        
        if x in cache:
            val = cache[x]
        else:
            calls += 1
            try:
                val = f(x)
                val = float(val)
            except Exception:
                val = float('nan')
            cache[x] = val
            
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_f:
                best_f = abs_val
                best_x = x
                if best_f <= 1e-8:
                    found = True
        return val

    # Initial evaluation
    fx0 = safe_f(x0)
    if found:
        return best_x

    def check_bracket_and_refine(xa, xb):
        """Check if [xa, xb] brackets a root, and if so refine it."""
        fa = cache.get(xa)
        fb = cache.get(xb)
        if fa is None:
            fa = safe_f(xa)
        if fb is None:
            fb = safe_f(xb)
        if not math.isfinite(fa) or not math.isfinite(fb):
            return None
        if fa * fb < 0:
            return refine_root((xa, xb), safe_f, cache)
        return None

    def refine_root(bracket, safe_f, cache):
        a, b = bracket
        if a > b:
            a, b = b, a
        
        fa = cache.get(a, safe_f(a))
        fb = cache.get(b, safe_f(b))
        
        if not math.isfinite(fa) or not math.isfinite(fb):
            return None
        
        # Try brentq first - fast convergence
        try:
            root = opt.brentq(safe_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=60)
            if math.isfinite(root):
                fr = safe_f(root)
                if math.isfinite(fr) and abs(fr) <= 1e-8:
                    return root
        except Exception:
            pass
        
        # Fallback: Bisection
        for _ in range(50):
            mid = (a + b) / 2.0
            fm = safe_f(mid)
            if not math.isfinite(fm):
                break
            if abs(fm) <= 1e-8:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        
        # Return best candidate
        candidates = [a, b]
        if 'mid' in locals():
            candidates.append(mid)
        best_local = min(candidates, key=lambda x: abs(cache.get(x, safe_f(x))))
        if abs(cache.get(best_local, safe_f(best_local))) <= 1e-8:
            return best_local
        return None

    # Phase 1: Fine local scan around x0
    # Scan +/- 1.0 with step 0.05 (40 points each direction = 80 calls max)
    h = 0.05
    for direction in [1, -1]:
        prev_x = x0
        prev_fx = fx0
        for i in range(1, 21):  # 10 steps = 0.5 range each direction first
            x_new = x0 + direction * i * h
            fx_new = safe_f(x_new)
            if found:
                return best_x
            if math.isfinite(prev_fx) and math.isfinite(fx_new) and prev_fx * fx_new < 0:
                result = refine_root((prev_x, x_new), safe_f, cache)
                if result is not None:
                    return result
            prev_x = x_new
            prev_fx = fx_new

    # Phase 2: Medium range scan
    # Continue from where we left off, expand to +/- 5.0
    for direction in [1, -1]:
        # Start from the frontier of phase 1
        start_i = 21
        prev_x = x0 + direction * 20 * h
        prev_fx = cache.get(prev_x, safe_f(prev_x))
        for i in range(start_i, int(5.0/h) + 1):
            x_new = x0 + direction * i * h
            fx_new = safe_f(x_new)
            if found:
                return best_x
            if math.isfinite(prev_fx) and math.isfinite(fx_new) and prev_fx * fx_new < 0:
                result = refine_root((prev_x, x_new), safe_f, cache)
                if result is not None:
                    return result
            prev_x = x_new
            prev_fx = fx_new

    # Phase 3: Coarser wider scan
    # +/- 50.0 with step 0.5
    h2 = 0.5
    for direction in [1, -1]:
        prev_x = x0 + direction * 5.0
        prev_fx = cache.get(prev_x, safe_f(prev_x))
        for i in range(1, int(45.0/h2) + 1):
            x_new = x0 + direction * (5.0 + i * h2)
            fx_new = safe_f(x_new)
            if found:
                return best_x
            if math.isfinite(prev_fx) and math.isfinite(fx_new) and prev_fx * fx_new < 0:
                result = refine_root((prev_x, x_new), safe_f, cache)
                if result is not None:
                    return result
            prev_x = x_new
            prev_fx = fx_new

    # Phase 4: Very wide scan
    # +/- 500.0 with step 5.0
    h3 = 5.0
    for direction in [1, -1]:
        prev_x = x0 + direction * 50.0
        prev_fx = cache.get(prev_x, safe_f(prev_x))
        for i in range(1, int(450.0/h3) + 1):
            x_new = x0 + direction * (50.0 + i * h3)
            fx_new = safe_f(x_new)
            if found:
                return best_x
            if math.isfinite(prev_fx) and math.isfinite(fx_new) and prev_fx * fx_new < 0:
                result = refine_root((prev_x, x_new), safe_f, cache)
                if result is not None:
                    return result
            prev_x = x_new
            prev_fx = fx_new

    # If no bracket found, return best_x
    return best_x
