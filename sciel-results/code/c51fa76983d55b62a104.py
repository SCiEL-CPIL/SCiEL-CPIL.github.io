"""Root finder for 3 fixed smooth polynomial equations."""
import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Uses a hybrid strategy:
    1. Quick bracket search with minimal evaluations.
    2. Fast Secant/Bisection hybrid with strict call budget.
    3. Track best point to always return the closest root found.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit
    TOL = 1e-9  # internal tolerance, stricter than required 1e-8
    MAX_CALLS = 495  # stay under 500 per equation

    call_count = 0
    best_x = float(x0)
    best_f = float('inf')
    best_abs = float('inf')

    def _check_time():
        return time.time() < deadline

    def _safe_f(x_val):
        nonlocal call_count, best_x, best_f, best_abs
        if call_count >= MAX_CALLS:
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_f = val
                best_x = x_val
            return val
        except Exception:
            call_count += 1
            return None

    x = float(x0)
    fx = _safe_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Phase 1: Find a bracket [a, b] with sign change.
    # Strategy: try small steps first (polynomial roots are often near x0),
    # then expand geometrically.
    a, b, fa, fb = None, None, None, None
    
    # List of steps to try. This covers a wide range without excessive calls.
    # We check both left and right for each step.
    steps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    
    for s in steps:
        if not _check_time() or call_count >= MAX_CALLS - 10:
            break

        # Try right
        xr = x + s
        fr = _safe_f(xr)
        if fr is None:
            continue
        if abs(fr) <= TOL:
            return xr
        if fx is not None and fx * fr < 0:
            a, b, fa, fb = x, xr, fx, fr
            break

        # Try left
        xl = x - s
        fl = _safe_f(xl)
        if fl is None:
            continue
        if abs(fl) <= TOL:
            return xl
        if fx is not None and fx * fl < 0:
            a, b, fa, fb = xl, x, fl, fx
            break

    # If no bracket found with fixed steps, try a geometric expansion
    if a is None:
        step = 1.0
        for _ in range(20):
            if not _check_time() or call_count >= MAX_CALLS - 10:
                break

            xr = x + step
            fr = _safe_f(xr)
            if fr is None:
                step *= 2.0
                continue
            if abs(fr) <= TOL:
                return xr
            if fx is not None and fx * fr < 0:
                a, b, fa, fb = x, xr, fx, fr
                break

            xl = x - step
            fl = _safe_f(xl)
            if fl is None:
                step *= 2.0
                continue
            if abs(fl) <= TOL:
                return xl
            if fx is not None and fx * fl < 0:
                a, b, fa, fb = xl, x, fl, fx
                break

            step *= 2.0
            if step > 1e8:
                break

    # Phase 2: If bracket found, use hybrid Secant-Bisection
    if a is not None and b is not None and fa is not None and fb is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb

        # Secant points: start with the endpoints
        x_prev2, f_prev2 = x_lo, f_lo
        x_prev1, f_prev1 = x_hi, f_hi

        for _ in range(50):
            if not _check_time() or call_count >= MAX_CALLS - 5:
                break

            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi

            # Midpoint for bisection fallback
            x_mid = (x_lo + x_hi) / 2.0
            f_mid = _safe_f(x_mid)
            if f_mid is None:
                break
            if abs(f_mid) <= TOL:
                return x_mid

            # Try secant step using the two most recent points
            x_new = None
            f_new = None

            if f_prev1 != f_prev2:
                # Secant formula: x_new = x_prev1 - f_prev1 * (x_prev1 - x_prev2) / (f_prev1 - f_prev2)
                x_sec = x_prev1 - f_prev1 * (x_prev1 - x_prev2) / (f_prev1 - f_prev2)
                if x_lo < x_sec < x_hi:
                    f_sec = _safe_f(x_sec)
                    if f_sec is not None:
                        if abs(f_sec) <= TOL:
                            return x_sec
                        x_new = x_sec
                        f_new = f_sec
                        # Update bracket
                        if f_sec * f_lo < 0:
                            x_hi = x_sec
                            f_hi = f_sec
                        else:
                            x_lo = x_sec
                            f_lo = f_sec
                        # Update secant points
                        x_prev2, f_prev2 = x_prev1, f_prev1
                        x_prev1, f_prev1 = x_sec, f_sec
                        continue

            # Fallback: bisection
            if f_mid * f_lo < 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid

            # Update secant points: keep the two points with different signs
            # that are closest to the root (or just the new midpoint and one endpoint)
            if f_lo * f_mid < 0:
                # Root is between x_lo and x_mid
                x_prev2, f_prev2 = x_prev1, f_prev1
                x_prev1, f_prev1 = x_mid, f_mid
            else:
                # Root is between x_mid and x_hi
                x_prev2, f_prev2 = x_prev1, f_prev1
                x_prev1, f_prev1 = x_mid, f_mid

        # Return the best endpoint or best tracked
        if abs(f_lo) < abs(f_hi):
            return x_lo
        else:
            return x_hi

    # Phase 3: No bracket found. Use Newton's method with finite difference.
    # For polynomials, Newton converges quadratically once close.
    x_curr = float(x0)
    f_curr = fx

    for _ in range(30):
        if not _check_time() or call_count >= MAX_CALLS - 5:
            break

        if abs(f_curr) <= TOL:
            return x_curr

        # Finite difference derivative with adaptive step
        h = 1e-8 * max(1.0, abs(x_curr))
        fx_plus = _safe_f(x_curr + h)
        if fx_plus is None:
            break

        d = (fx_plus - f_curr) / h
        if d == 0 or not math.isfinite(d):
            # Try smaller step
            h = 1e-10 * max(1.0, abs(x_curr))
            fx_plus = _safe_f(x_curr + h)
            if fx_plus is None:
                break
            d = (fx_plus - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break

        x_new = x_curr - f_curr / d
        if not math.isfinite(x_new):
            break

        # Damping: limit step to avoid wild overshoots
        max_step = 2.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        # If we're making progress, continue; otherwise try bracketing
        if abs(f_new) < abs(f_curr) * 0.9:
            x_curr = x_new
            f_curr = f_new
        else:
            # Stagnation: try to bracket around x_curr
            # Quick check: x_curr - h, x_curr, x_curr + h
            # We already have f_curr.
            # Try x_curr - 1.0 and x_curr + 1.0 if not tried
            # For simplicity, just break and return best
            break

    return best_x

# EVOLVE-BLOCK-END
