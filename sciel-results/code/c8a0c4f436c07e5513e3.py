import time
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0
    TOL = 1e-12

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    try:
        f0 = f(x0)
    except Exception:
        f0 = float('nan')

    best_x = x0
    best_val = abs(f0) if f0 == f0 else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx == fx:
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    if f0 == f0 and abs(f0) <= TOL:
        return x0

    a = b = None
    fa = fb = None

    base = max(1.0, abs(x0))

    # Phase 1: Coarse outward scan to estimate scale and find easy brackets
    # Uses ~40 calls
    step = base * 1e-2
    x_prev = x0
    f_prev = f0
    x_curr = x0
    for _ in range(40):
        if not elapsed_ok():
            break
        x_next = x_curr + step
        try:
            f_next = f(x_next)
        except Exception:
            f_next = float('nan')
        track(x_next, f_next)
        
        if f_next == f_next and f_prev == f_prev and f_prev * f_next < 0:
            a, b, fa, fb = x_prev, x_next, f_prev, f_next
            break
        
        x_prev, f_prev = x_next, f_next
        x_curr = x_next
        step *= 1.5

    # Phase 2: Dense scan over a range determined by Phase 1
    # Uses ~400 calls
    if a is None:
        # Use the current step as the radius for dense scan
        # step is now roughly base * 1e-2 * 1.5^40 which is very large
        # Let's cap the radius to something reasonable to avoid floating point issues or huge ranges
        # But we want to catch distant roots.
        # Let's use the range [x0 - step, x0 + step]
        R = step
        # If R is too small (e.g. if loop didn't run much), ensure minimum
        if R < base:
            R = base
            
        n_points = 400
        left = x0 - R
        right = x0 + R
        
        # To save calls, we can skip points if we are confident, but let's just do it.
        # We already have f(x0). We can reuse it if it falls in the grid? Unlikely to align.
        # Just do a fresh scan.
        
        prev_x = left
        try:
            prev_f = f(prev_x)
        except Exception:
            prev_f = float('nan')
        track(prev_x, prev_f)
        
        for i in range(1, n_points + 1):
            if not elapsed_ok():
                break
            x_i = left + (right - left) * i / n_points
            try:
                f_i = f(x_i)
            except Exception:
                f_i = float('nan')
            track(x_i, f_i)
            
            if f_i == f_i and prev_f == f_i and prev_f * f_i < 0:
                a, b, fa, fb = prev_x, x_i, prev_f, f_i
                break
            prev_x, prev_f = x_i, f_i

    if a is not None:
        try:
            root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            f_root = f(root)
            if f_root == f_root and abs(f_root) <= 1e-8:
                return root
            else:
                track(root, f_root)
                if best_val <= 1e-8:
                    return best_x
        except Exception:
            pass

    # Fallback
    try:
        x = x0
        fx = f0
        for _ in range(50):
            if not elapsed_ok():
                break
            if fx == fx and abs(fx) <= TOL:
                return x
            h = 1e-6 * max(1.0, abs(x))
            try:
                d = (f(x + h) - fx) / h
            except Exception:
                break
            if d == 0 or d != d:
                break
            xn = x - fx / d
            if xn != xn or xn == float('inf') or xn == float('-inf'):
                break
            try:
                fn = f(xn)
            except Exception:
                break
            track(xn, fn)
            if fn == fn and abs(fn) <= TOL:
                return xn
            if abs(xn - x) > 1e6 * max(1.0, abs(x)):
                xn = 0.5 * (x + xn)
            x, fx = xn, fn
    except Exception:
        pass

    if best_x == best_x and abs(best_val) < float('inf'):
        return best_x
    return x0
# EVOLVE-BLOCK-END
