"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding for ill-conditioned equations.
    Strategy:
      1. Try a damped Newton/secant phase from x0.
      2. If not converged, attempt global bracketing via expanding intervals.
      3. Use Brent's method (via scipy) on the bracket if found.
      4. If no bracket, use a local minimization of |f(x)| via golden-section or Nelder-Mead.
      5. Final polish with high-precision Newton/secant.
    Deadline guard: return best x found before 145s.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    x0 = float(x0)
    best_x = x0
    best_f = f(best_x)
    best_abs = abs(best_f)
    eval_count = 1

    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs, eval_count
        ax = abs(fx)
        if ax < best_abs:
            best_x = x
            best_f = fx
            best_abs = ax

    update_best(best_x, best_f)

    # Helper: safe function evaluation
    def safe_f(x):
        nonlocal eval_count
        if not time_ok():
            return None
        try:
            val = f(x)
            eval_count += 1
            if not math.isfinite(val):
                return None
            return val
        except:
            eval_count += 1
            return None

    # Phase 1: Damped Newton / Secant
    x = x0
    fx = best_f
    x_prev = x0 + 1e-4 * max(1.0, abs(x0))
    f_prev = safe_f(x_prev)
    if f_prev is None:
        f_prev = fx
        x_prev = x0

    for it in range(200):
        if not time_ok():
            break
        if abs(fx) <= 1e-8:
            return x
        update_best(x, fx)

        # Compute derivative via secant
        if abs(x - x_prev) > 1e-15:
            d = (fx - f_prev) / (x - x_prev)
        else:
            h = 1e-6 * max(1.0, abs(x))
            fx_h = safe_f(x + h)
            if fx_h is None:
                break
            d = (fx_h - fx) / h

        if d == 0 or not math.isfinite(d):
            break

        # Damped Newton step
        step = fx / d
        # Limit step size
        max_step = 1.0 + abs(x)
        if abs(step) > max_step:
            step = max_step * (1.0 if step > 0 else -1.0)

        x_new = x - step
        if not math.isfinite(x_new):
            break

        fx_new = safe_f(x_new)
        if fx_new is None:
            break

        # Accept or reject (simple damping)
        if abs(fx_new) < abs(fx) or it == 0:
            x_prev = x
            f_prev = fx
            x = x_new
            fx = fx_new
        else:
            # Try half step
            x_half = x - step / 2
            fx_half = safe_f(x_half)
            if fx_half is not None and abs(fx_half) < abs(fx):
                x_prev = x
                f_prev = fx
                x = x_half
                fx = fx_half
            else:
                # Try smaller step
                x_small = x - step / 4
                fx_small = safe_f(x_small)
                if fx_small is not None and abs(fx_small) < abs(fx):
                    x_prev = x
                    f_prev = fx
                    x = x_small
                    fx = fx_small
                else:
                    break

    if abs(fx) <= 1e-8:
        return x
    update_best(x, fx)

    # Phase 2: Try to find a bracket
    # Expand from x0 in both directions
    a = x0
    fa = best_f
    b = x0
    fb = best_f

    # Try expanding to the right
    scale = 1.0
    found_bracket = False
    for i in range(60):
        if not time_ok():
            break
        b = x0 + scale
        fb = safe_f(b)
        if fb is None:
            scale *= 2.0
            continue
        update_best(b, fb)
        if fa * fb < 0:
            found_bracket = True
            break
        a = b
        fa = fb
        scale *= 2.0
        if abs(scale) > 1e12:
            break

    if not found_bracket:
        # Try expanding to the left
        a = x0
        fa = best_f
        scale = 1.0
        for i in range(60):
            if not time_ok():
                break
            b = x0 - scale
            fb = safe_f(b)
            if fb is None:
                scale *= 2.0
                continue
            update_best(b, fb)
            if fa * fb < 0:
                found_bracket = True
                break
            a = b
            fa = fb
            scale *= 2.0
            if abs(scale) > 1e12:
                break

    if found_bracket and time_ok():
        # Use scipy's brentq
        try:
            from scipy.optimize import brentq
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if time_ok():
                fr = f(root)
                update_best(root, fr)
                if abs(fr) <= 1e-8:
                    return root
        except:
            pass

    # Phase 3: Local minimization of |f(x)| using golden-section search
    if time_ok():
        # Find a local minimum of |f| near x0
        # Use a window around the best point
        center = best_x
        window = max(1.0, abs(center))
        lo = center - window
        hi = center + window

        # First evaluate endpoints and center
        f_lo = safe_f(lo)
        f_hi = safe_f(hi)
        f_c = safe_f(center)

        if f_lo is not None:
            update_best(lo, f_lo)
        if f_hi is not None:
            update_best(hi, f_hi)
        if f_c is not None:
            update_best(center, f_c)

        # Golden-section search for minimum of |f(x)|
        gr = (math.sqrt(5) - 1) / 2  # golden ratio conjugate
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)

        for i in range(80):
            if not time_ok():
                break
            f1 = safe_f(x1)
            f2 = safe_f(x2)

            if f1 is None or f2 is None:
                break

            update_best(x1, f1)
            update_best(x2, f2)

            if abs(f1) < abs(f2):
                hi = x2
                x2 = x1
                x1 = hi - gr * (hi - lo)
            else:
                lo = x1
                x1 = x2
                x2 = lo + gr * (hi - lo)

            if abs(hi - lo) < 1e-10 * max(1.0, abs(center)):
                break

        # Polish the best point with Newton/secant
        x = best_x
        fx = best_f
        x_prev = x + 1e-4 * max(1.0, abs(x))
        f_prev = safe_f(x_prev)
        if f_prev is None:
            f_prev = fx

        for it in range(50):
            if not time_ok():
                break
            if abs(fx) <= 1e-12:
                break
            if abs(x - x_prev) > 1e-15:
                d = (fx - f_prev) / (x - x_prev)
            else:
                h = 1e-8 * max(1.0, abs(x))
                fx_h = safe_f(x + h)
                if fx_h is None:
                    break
                d = (fx_h - fx) / h

            if d == 0 or not math.isfinite(d):
                break

            step = fx / d
            max_step = 0.1 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = max_step * (1.0 if step > 0 else -1.0)

            x_new = x - step
            if not math.isfinite(x_new):
                break

            fx_new = safe_f(x_new)
            if fx_new is None:
                break

            if abs(fx_new) < abs(fx):
                x_prev = x
                f_prev = fx
                x = x_new
                fx = fx_new
            else:
                break

        update_best(x, fx)

    return best_x
# EVOLVE-BLOCK-END
