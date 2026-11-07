"""Root finder for 3 fixed smooth polynomial equations."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Optimized for smooth polynomials with minimal function calls.
    """
    start_time = time.time()
    deadline = start_time + 118.0
    TOL = 1e-9
    MAX_CALLS = 498

    call_count = 0
    best_x = float(x0)
    best_abs = float('inf')

    def _check_time():
        return time.time() < deadline

    def _safe_f(x_val):
        nonlocal call_count, best_x, best_abs
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

    # Phase 1: Find bracket with sign change
    # Use adaptive geometric expansion, checking both directions
    a, b, fa, fb = None, None, None, None

    # Try a sequence of offsets: small to large, both sides
    # For polynomials, roots are often within reasonable distance
    offsets = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
               100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]

    for s in offsets:
        if not _check_time() or call_count >= MAX_CALLS - 8:
            break

        # Check right
        xr = x + s
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx * fr < 0:
                a, b, fa, fb = x, xr, fx, fr
                break

        # Check left
        xl = x - s
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx * fl < 0:
                a, b, fa, fb = xl, x, fl, fx
                break

    # If no bracket with fixed offsets, try geometric expansion
    if a is None:
        step = 1.0
        for _ in range(25):
            if not _check_time() or call_count >= MAX_CALLS - 8:
                break

            xr = x + step
            fr = _safe_f(xr)
            if fr is not None:
                if abs(fr) <= TOL:
                    return xr
                if fx * fr < 0:
                    a, b, fa, fb = x, xr, fx, fr
                    break

            xl = x - step
            fl = _safe_f(xl)
            if fl is not None:
                if abs(fl) <= TOL:
                    return xl
                if fx * fl < 0:
                    a, b, fa, fb = xl, x, fl, fx
                    break

            step *= 2.0
            if step > 1e10:
                break

    # Phase 2: Solve with secant method if bracket found
    if a is not None and b is not None and fa is not None and fb is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb

        # Secant iteration: 1 new function call per step
        x_prev2, f_prev2 = x_lo, f_lo
        x_prev1, f_prev1 = x_hi, f_hi

        for _ in range(40):
            if not _check_time() or call_count >= MAX_CALLS - 3:
                break

            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi

            # Secant step
            if f_prev1 != f_prev2:
                x_sec = x_prev1 - f_prev1 * (x_prev1 - x_prev2) / (f_prev1 - f_prev2)
                if math.isfinite(x_sec) and x_lo < x_sec < x_hi:
                    f_sec = _safe_f(x_sec)
                    if f_sec is not None:
                        if abs(f_sec) <= TOL:
                            return x_sec
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

            # Bisection fallback
            x_mid = (x_lo + x_hi) / 2.0
            f_mid = _safe_f(x_mid)
            if f_mid is None:
                break
            if abs(f_mid) <= TOL:
                return x_mid

            if f_mid * f_lo < 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid

            x_prev2, f_prev2 = x_prev1, f_prev1
            x_prev1, f_prev1 = x_mid, f_mid

        # Return best of endpoints
        if abs(f_lo) < abs(f_hi):
            return x_lo
        else:
            return x_hi

    # Phase 3: No bracket found - use damped Newton
    x_curr = float(x0)
    f_curr = fx

    for _ in range(25):
        if not _check_time() or call_count >= MAX_CALLS - 5:
            break

        if abs(f_curr) <= TOL:
            return x_curr

        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x_curr))
        fx_plus = _safe_f(x_curr + h)
        if fx_plus is None:
            break

        d = (fx_plus - f_curr) / h
        if d == 0 or not math.isfinite(d):
            h = 1e-9 * max(1.0, abs(x_curr))
            fx_plus2 = _safe_f(x_curr + h)
            if fx_plus2 is None:
                break
            d = (fx_plus2 - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break

        x_new = x_curr - f_curr / d
        if not math.isfinite(x_new):
            break

        # Damp step
        max_step = 3.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        if abs(f_new) < abs(f_curr) * 0.95:
            x_curr = x_new
            f_curr = f_new
        else:
            break

    return best_x

# EVOLVE-BLOCK-END
