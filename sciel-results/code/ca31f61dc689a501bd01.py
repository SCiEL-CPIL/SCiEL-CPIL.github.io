"""Hybrid root-finder: multi-scale sign-change scan + Brent-style refinement + Newton polish."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 148.0
    HARD_TOL = 1e-8
    MAX_CALLS = 485

    call_count = 0

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            call_count += 1
            return float('nan')

    best_x = x0
    best_fx = None

    def track(x, fx):
        nonlocal best_x, best_fx
        if fx != float('nan') and math.isfinite(fx):
            if best_fx is None or abs(fx) < abs(best_fx):
                best_x = x
                best_fx = fx

    # Cache with simple dict
    eval_cache = {}

    def get_f(x):
        key = round(x, 15)
        if key in eval_cache:
            return eval_cache[key]
        val = safe_f(x)
        eval_cache[key] = val
        track(x, val)
        return val

    def brent_refine(a, b, fa, fb, max_iters=80):
        """Brent's method: combines bisection, secant, and inverse quadratic interpolation."""
        if not elapsed_ok() or call_count >= MAX_CALLS - 3:
            return None
        if abs(fa) <= HARD_TOL:
            return a
        if abs(fb) <= HARD_TOL:
            return b
        if fa * fb > 0:
            return None

        # Ensure |fa| <= |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c = a
        fc = fa
        d = b - a
        e = d
        A = B = 0.0

        for _ in range(max_iters):
            if not elapsed_ok() or call_count >= MAX_CALLS - 2:
                break

            # Check for convergence
            if abs(fb) <= HARD_TOL:
                return b
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                return b if abs(fb) < abs(fa) else a

            # Inverse quadratic interpolation or secant
            if A != B:
                # A and B are from previous iteration
                pass

            if abs(e) > 1e-14 * max(1.0, abs(b)):
                # Secant step
                A = fc / fb
                B = fa / fb
                s = b - (b - a) * A * B / (A - B)
            else:
                # Bisection
                s = (a + b) / 2.0

            # Check if s is acceptable
            if (s <= (a + b) / 2.0 - 1e-15 * max(1.0, abs(a)) or
                    s >= (a + b) / 2.0 + 1e-15 * max(1.0, abs(b)) or
                    abs(s - b) >= 0.5 * abs(b - a) or
                    (abs(e) <= 1e-14 * max(1.0, abs(b)) and A == B)):
                s = (a + b) / 2.0
                e = d

            d = b - a
            fs = get_f(s)
            if fs == float('nan'):
                break

            if abs(fs) <= HARD_TOL:
                return s

            if fa * fs < 0:
                b, fb = s, fs
            else:
                a, fa = s, fs
                if abs(fa) <= abs(fb):
                    b, fb = s, fs

            # Swap to maintain |fa| <= |fb|
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa

            e = d
            d = b - a

        # Return best
        if abs(fb) < abs(fa):
            return b
        return a

    def newton_polish(x_start, max_iters=30):
        """Newton's method with secant fallback for final polish."""
        if not elapsed_ok() or call_count >= MAX_CALLS - 5:
            return None

        x = x_start
        fx = get_f(x)
        if fx == float('nan'):
            return None
        if abs(fx) <= HARD_TOL:
            return x

        # Secant method for derivative-free Newton-like convergence
        h = max(1e-10, abs(x) * 1e-10)
        x1 = x + h
        fx1 = get_f(x1)
        if fx1 == float('nan'):
            x1 = x - h
            fx1 = get_f(x1)
            if fx1 == float('nan'):
                return None

        for _ in range(max_iters):
            if not elapsed_ok() or call_count >= MAX_CALLS - 2:
                break
            if abs(fx) <= HARD_TOL:
                return x
            if abs(fx1 - fx) < 1e-16:
                break
            x_new = x - fx * (x - x1) / (fx - fx1)
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
                break
            x1, x = x, x_new
            fx1, fx = fx, get_f(x)
            if fx == float('nan'):
                break

        if abs(fx) <= HARD_TOL:
            return x
        return None

    # Evaluate starting point
    f0 = get_f(x0)
    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    # Phase 1: Multi-scale sign-change scanning
    # Try logarithmically spaced intervals to find a bracket
    base = max(1e-8, abs(x0) * 1e-6)
    # Scales to try: from very fine to very coarse
    scales = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    found_bracket = None

    for scale in scales:
        if not elapsed_ok() or call_count >= MAX_CALLS - 20:
            break

        interval = base * scale
        if interval <= 0:
            continue

        # Scan in both directions from x0
        for direction in [1, -1]:
            if not elapsed_ok() or call_count >= MAX_CALLS - 20:
                break

            prev_x = x0
            prev_fx = f0

            # Scan up to 200 points in this direction
            for i in range(1, 201):
                if not elapsed_ok() or call_count >= MAX_CALLS - 20:
                    break

                x_i = x0 + direction * i * interval
                fx_i = get_f(x_i)
                if fx_i == float('nan'):
                    # Try to skip this point
                    prev_x = x_i
                    prev_fx = fx_i
                    continue

                if abs(fx_i) <= HARD_TOL:
                    return x_i

                # Check for sign change
                if prev_fx != float('nan') and prev_fx * fx_i < 0:
                    found_bracket = (prev_x, x_i, prev_fx, fx_i)
                    break

                prev_x = x_i
                prev_fx = fx_i

            if found_bracket is not None:
                break

        if found_bracket is not None:
            break

    # Phase 2: If bracket found, refine with Brent
    if found_bracket is not None:
        a, b, fa, fb = found_bracket
        result = brent_refine(a, b, fa, fb)
        if result is not None:
            # Polish with Newton
            polished = newton_polish(result)
            if polished is not None:
                return polished
            return result

    # Phase 3: No bracket found. Try to find minimum of |f| via local search
    # Use a golden-section-like search around x0 with expanding window
    if best_fx is not None and abs(best_fx) > HARD_TOL:
        # Try expanding window search
        window = max(1e-8, abs(x0) * 1e-4)
        for _ in range(8):
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break

            # Sample 50 points in [-window, window] around best_x
            n_pts = 50
            prev_x = best_x - window
            prev_fx = get_f(prev_x)

            for i in range(1, n_pts + 1):
                if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                    break
                x_i = best_x - window + 2 * window * i / n_pts
                fx_i = get_f(x_i)
                if fx_i == float('nan'):
                    prev_x, prev_fx = x_i, fx_i
                    continue

                if abs(fx_i) <= HARD_TOL:
                    return x_i

                if prev_fx != float('nan') and prev_fx * fx_i < 0:
                    result = brent_refine(prev_x, x_i, prev_fx, fx_i)
                    if result is not None:
                        polished = newton_polish(result)
                        if polished is not None:
                            return polished
                        return result

                prev_x, prev_fx = x_i, fx_i

            # Expand window
            window *= 10.0

    # Phase 4: Last resort - try Newton from best_x
    if best_fx is not None and abs(best_fx) > HARD_TOL:
        result = newton_polish(best_x)
        if result is not None:
            return result

    # Return best found
    if best_fx is not None and math.isfinite(best_fx):
        return best_x
    return x0
# EVOLVE-BLOCK-END
