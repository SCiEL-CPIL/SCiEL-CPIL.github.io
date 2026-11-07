"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for smooth polynomial equations.
    Hybrid: bracket + bisection/secant, Newton polish.
    """
    x0 = float(x0)
    TOL = 1e-10  # internal tolerance, stricter than 1e-8 requirement
    REQUIRED_TOL = 1e-8
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s kill
    MAX_CALLS = 490
    call_count = 0
    best_x = x0
    best_f = float('inf')

    def _check_time():
        return time.time() < deadline

    def _safe_f(x_val):
        nonlocal call_count, best_x, best_f
        if call_count >= MAX_CALLS:
            return None
        if not _check_time():
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            if abs(val) < abs(best_f):
                best_f = val
                best_x = x_val
            return val
        except Exception:
            call_count += 1
            return None

    def _newton_polish(x, fx, max_iters=15):
        """Newton's method with finite-diff derivative, for final polish."""
        nonlocal call_count, best_x, best_f
        for _ in range(max_iters):
            if not _check_time() or call_count >= MAX_CALLS - 2:
                break
            if abs(fx) <= REQUIRED_TOL:
                return x
            # Try several step sizes for the derivative
            for h in [1e-8 * max(1.0, abs(x)), 1e-7 * max(1.0, abs(x)), 1e-6 * max(1.0, abs(x))]:
                fx_plus = _safe_f(x + h)
                if fx_plus is None:
                    break
                d = (fx_plus - fx) / h
                if d == 0 or not math.isfinite(d):
                    continue
                x_new = x - fx / d
                if not math.isfinite(x_new):
                    continue
                # Damping
                if abs(x_new - x) > 5.0 * max(1.0, abs(x)):
                    x_new = x + 0.5 * (x_new - x)
                fx_new = _safe_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                    if abs(fx) < abs(best_f):
                        best_x, best_f = x, fx
                    if abs(fx) <= REQUIRED_TOL:
                        return x
                    break  # success, stop trying other h
                else:
                    # Try smaller h
                    continue
            else:
                # All h tried without improvement
                break
        return x

    # Phase 1: Evaluate at x0
    f0 = _safe_f(x0)
    if f0 is None:
        return best_x
    if abs(f0) <= TOL:
        return x0

    # Phase 2: Try to find a bracket around x0
    # Strategy: try small offsets first (cheap, catches nearby roots), then geometric expansion
    a, b, fa, fb = None, None, None, None

    # Try a set of small offsets around x0
    base_step = max(1e-4, abs(x0) * 1e-4)
    for offset in [0.1, 0.01, 1.0, 0.1, 10.0, 100.0, 1000.0]:
        step = base_step * offset
        if not _check_time():
            break
        # Right
        xr = x0 + step
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if f0 * fr < 0:
                a, b = x0, xr
                fa, fb = f0, fr
                break
        # Left
        xl = x0 - step
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if f0 * fl < 0:
                a, b = xl, x0
                fa, fb = fl, f0
                break

    # If no bracket from small offsets, do geometric expansion
    if a is None:
        step = max(1.0, abs(x0)) * 0.5
        for _ in range(25):
            if not _check_time() or call_count >= MAX_CALLS - 5:
                break
            # Right expansion
            xr = x0 + step
            fr = _safe_f(xr)
            if fr is not None:
                if abs(fr) <= TOL:
                    return xr
                if f0 * fr < 0:
                    a, b = x0, xr
                    fa, fb = f0, fr
                    break
            # Left expansion
            xl = x0 - step
            fl = _safe_f(xl)
            if fl is not None:
                if abs(fl) <= TOL:
                    return xl
                if f0 * fl < 0:
                    a, b = xl, x0
                    fa, fb = fl, f0
                    break
            step *= 2.0
            if step > 1e12:
                break

    # Phase 3: If bracket found, use hybrid bisection + secant
    if a is not None and b is not None and fa is not None and fb is not None:
        # Ensure fa < 0 < fb or vice versa (sign change)
        if fa * fb > 0:
            # Shouldn't happen, but handle gracefully
            a, b, fa, fb = b, a, fb, fa

        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb

        # For secant: track last two points
        x_prev, f_prev = x_hi, f_hi
        x_curr, f_curr = x_lo, f_lo

        for _ in range(80):
            if not _check_time() or call_count >= MAX_CALLS - 3:
                break
            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi

            # Midpoint (bisection)
            x_mid = (x_lo + x_hi) / 2.0
            f_mid = _safe_f(x_mid)
            if f_mid is None:
                break
            if abs(f_mid) <= TOL:
                return x_mid

            # Try secant step if we have two points
            success = False
            if f_prev is not None and f_curr is not None and f_prev != f_curr:
                x_sec = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                if math.isfinite(x_sec) and x_lo < x_sec < x_hi:
                    f_sec = _safe_f(x_sec)
                    if f_sec is not None:
                        if abs(f_sec) <= TOL:
                            return x_sec
                        # Update bracket
                        if f_sec * f_lo < 0:
                            x_hi, f_hi = x_sec, f_sec
                        else:
                            x_lo, f_lo = x_sec, f_sec
                        # Update secant points
                        x_prev, f_prev = x_curr, f_curr
                        x_curr, f_curr = x_sec, f_sec
                        success = True

            if not success:
                # Bisection update
                if f_mid * f_lo < 0:
                    x_hi, f_hi = x_mid, f_mid
                else:
                    x_lo, f_lo = x_mid, f_mid
                x_prev, f_prev = x_curr, f_curr
                x_curr, f_curr = x_mid, f_mid

            # Convergence check
            if abs(x_hi - x_lo) < 1e-15 * max(1.0, abs(x_mid)):
                break

        # Check endpoints after loop
        if abs(f_lo) <= TOL:
            return x_lo
        if abs(f_hi) <= TOL:
            return x_hi
        # Return the better endpoint
        if abs(f_lo) < abs(f_hi):
            return x_lo
        else:
            return x_hi

    # Phase 4: No bracket found — use Newton's method
    x = x0
    fx = f0
    for i in range(30):
        if not _check_time() or call_count >= MAX_CALLS - 5:
            break
        if abs(fx) <= TOL:
            return x

        # Finite difference derivative with multiple step sizes
        found_step = False
        for h in [1e-7 * max(1.0, abs(x)), 1e-6 * max(1.0, abs(x)), 1e-8 * max(1.0, abs(x))]:
            fx_plus = _safe_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                continue
            x_new = x - fx / d
            if not math.isfinite(x_new):
                continue
            # Damping
            max_step = 2.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + max_step * (1.0 if x_new > x else -1.0)
            fx_new = _safe_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
                found_step = True
                if abs(fx) <= TOL:
                    return x
                break
            else:
                # Newton diverging, try smaller h
                continue

        if not found_step:
            break

    if abs(fx) <= TOL:
        return x

    # Phase 5: Secant method as fallback
    h = 1e-6 * max(1.0, abs(x))
    x2 = x + h
    f2 = _safe_f(x2)
    if f2 is None:
        x2 = x - h
        f2 = _safe_f(x2)

    if f2 is not None and f2 != fx:
        x1, f1 = x, fx
        for i in range(40):
            if not _check_time() or call_count >= MAX_CALLS - 3:
                break
            if abs(f1) <= TOL:
                return x1
            if abs(f2) <= TOL:
                return x2
            denom = f2 - f1
            if denom == 0 or not math.isfinite(denom):
                break
            x_new = x1 - f1 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                break
            f_new = _safe_f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new
            # Keep the point with smaller |f|
            if abs(f_new) < abs(f1):
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
            else:
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new

    # Phase 6: Final Newton polish on best point
    if abs(best_f) <= 1e-4:
        result = _newton_polish(best_x, best_f, max_iters=10)
        if abs(best_f) <= REQUIRED_TOL:
            return result

    # Return best point found
    return best_x
# EVOLVE-BLOCK-END
