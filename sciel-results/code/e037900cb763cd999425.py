"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 140.0  # Safety margin
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(val) and abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val

    def safeguarded_newton(x0, max_iters=50):
        """Newton's method with strict bracket maintenance and bisection fallback."""
        x = float(x0)
        h = 1e-6 * max(1.0, abs(x))
        fx = call_f(x)
        if fx is None:
            return None
        if abs(fx) <= TOL:
            return x
        
        # Try to establish a bracket around x
        # We want a bracket [a, b] containing x where f(a)*f(b) < 0
        # Start by expanding from x
        a, b = x - h, x + h
        fa = call_f(a)
        fb = call_f(b)
        
        # If we can't evaluate at a or b, try larger steps
        if fa is None or fb is None:
            for scale in [1e-4, 1e-2, 1e-1, 1.0, 10.0]:
                h2 = scale * max(1.0, abs(x))
                a, b = x - h2, x + h2
                fa = call_f(a)
                fb = call_f(b)
                if fa is not None and fb is not None:
                    break
            if fa is None or fb is None:
                return None
        
        # Check if we have a bracket
        has_bracket = False
        if fa * fb < 0:
            has_bracket = True
        elif fa * fx < 0:
            # Root between a and x
            b = x
            fb = fx
            has_bracket = True
        elif fb * fx < 0:
            # Root between x and b
            a = x
            fa = fx
            has_bracket = True
        
        if not has_bracket:
            return None
        
        # Now we have a bracket [a, b] with f(a)*f(b) < 0
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
                return None
            
            # Try Newton step from the point closer to the root (smaller |f|)
            if abs(fa) < abs(fb):
                # Newton from a
                # Estimate derivative using secant over [a, b]
                if abs(b - a) < 1e-15:
                    break
                d = (fb - fa) / (b - a)
                if abs(d) < 1e-15:
                    break
                x_new = a - fa / d
            else:
                # Newton from b
                if abs(b - a) < 1e-15:
                    break
                d = (fb - fa) / (b - a)
                if abs(d) < 1e-15:
                    break
                x_new = b - fb / d
            
            # Check if Newton step is within bracket
            if a < x_new < b:
                f_new = call_f(x_new)
                if f_new is None:
                    break
                if abs(f_new) <= TOL:
                    return x_new
                
                # Update bracket
                if fa * f_new < 0:
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
            else:
                # Newton step outside bracket, use bisection
                x_new = 0.5 * (a + b)
                f_new = call_f(x_new)
                if f_new is None:
                    break
                if abs(f_new) <= TOL:
                    return x_new
                
                if fa * f_new < 0:
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
            
            # Convergence check
            if abs(b - a) <= 1e-14 * max(1.0, abs(x_new)):
                # Return the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b
        return None

    def illinois(lo, hi, flo, fhi, max_iters=80):
        """Illinois method for guaranteed convergence within a bracket."""
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
                return None
            if flo * fhi > 0.0:
                break
            # Ensure |flo| < |fhi| for Illinois update
            if abs(flo) < abs(fhi):
                lo, hi = hi, lo
                flo, fhi = fhi, flo
            
            denom = fhi - flo
            if abs(denom) < 1e-300:
                mid = 0.5 * (lo + hi)
            else:
                mid = (flo * hi - fhi * lo) / denom
                if not (lo < mid < hi):
                    mid = 0.5 * (lo + hi)
            
            fm = call_f(mid)
            if fm is None:
                break
            if abs(fm) <= TOL:
                return mid
            
            if flo * fm < 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, 0.5 * flo
            
            if abs(hi - lo) <= 1e-14 * max(1.0, abs(mid)):
                return mid
        return None

    def find_bracket(x0, fx0, max_expansions=50):
        if fx0 is None:
            return None
        if abs(fx0) <= TOL:
            return (x0, x0, fx0, fx0)
        
        step = 0.1 * max(1.0, abs(x0))
        for direction in (1.0, -1.0):
            b = x0 + direction * step
            fb = call_f(b)
            if fb is None:
                continue
            if fx0 * fb < 0.0:
                if direction > 0:
                    return (x0, b, fx0, fb)
                else:
                    return (b, x0, fb, fx0)
            
            cur = b
            curf = fb
            for _ in range(max_expansions):
                if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
                    return None
                step *= 2.0
                nxt = x0 + direction * step
                fn = call_f(nxt)
                if fn is None:
                    break
                if curf * fn < 0.0:
                    if direction > 0:
                        return (cur, nxt, curf, fn)
                    else:
                        return (nxt, cur, fn, curf)
                cur, curf = nxt, fn
        return None

    # 1. Try safeguarded Newton from x0
    result = safeguarded_newton(x0)
    if result is not None: return result

    # 2. Try bracketing + Illinois from x0
    fx0 = call_f(float(x0))
    if fx0 is not None:
        bracket = find_bracket(float(x0), fx0)
        if bracket is not None:
            lo, hi, flo, fhi = bracket
            result = illinois(lo, hi, flo, fhi)
            if result is not None: return result

    # 3. Try safeguarded Newton from offset points
    offsets = [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0]
    for off in offsets:
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
            break
        x_try = float(x0) + off
        result = safeguarded_newton(x_try, max_iters=30)
        if result is not None: return result
        
        # Try bracketing from offset
        fx_try = call_f(x_try)
        if fx_try is not None:
            bracket = find_bracket(x_try, fx_try, max_expansions=20)
            if bracket is not None:
                lo, hi, flo, fhi = bracket
                result = illinois(lo, hi, flo, fhi, max_iters=40)
                if result is not None: return result

    # 4. Random restarts if budget allows
    if evals < MAX_EVALS - 50 and time.monotonic() < DEADLINE - 10:
        random.seed(42)
        for _ in range(10):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
                break
            scale = max(1.0, abs(float(x0)))
            x_try = float(x0) + random.uniform(-10*scale, 10*scale)
            
            fx_try = call_f(x_try)
            if fx_try is None:
                continue
            if abs(fx_try) <= TOL:
                return x_try
            
            result = safeguarded_newton(x_try, max_iters=15)
            if result is not None: return result
            
            bracket = find_bracket(x_try, fx_try, max_expansions=10)
            if bracket is not None:
                lo, hi, flo, fhi = bracket
                result = illinois(lo, hi, flo, fhi, max_iters=20)
                if result is not None: return result

    return best_x
# EVOLVE-BLOCK-END
