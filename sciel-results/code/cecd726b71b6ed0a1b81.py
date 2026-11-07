"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Brent-style root finder: IQUO + bisection hybrid on a bracket,
    with efficient bracket discovery for smooth polynomials.
    """
    start_time = time.time()
    deadline = start_time + 118.0
    TOL = 1e-8
    MAX_CALLS = 495

    call_count = 0
    best_x = float(x0)
    best_f = float('inf')

    def _check_time():
        return time.time() < deadline

    def _safe_f(x_val):
        nonlocal call_count, best_x, best_f
        if call_count >= MAX_CALLS:
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

    x = float(x0)
    fx = _safe_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Phase 1: Find a bracket [a, b] with sign change.
    a, b, fa, fb = None, None, None, None

    # Try adaptive steps from x0
    for s in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
              500.0, 1000.0, 2000.0, 5000.0, 10000.0]:
        if not _check_time() or call_count >= MAX_CALLS - 10:
            break

        xr = x + s
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx * fr < 0:
                a, b, fa, fb = x, xr, fx, fr
                break

        xl = x - s
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx * fl < 0:
                a, b, fa, fb = xl, x, fl, fx
                break

    # Geometric expansion if needed
    if a is None:
        step = 1.0
        for _ in range(15):
            if not _check_time() or call_count >= MAX_CALLS - 10:
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
            if step > 1e8:
                break

    # Phase 2: Brent's method (IQUO + bisection) on the bracket
    if a is not None and b is not None and fa is not None and fb is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Initialize: c = b, d = b - a (last step size)
        # Track previous point for IQUO
        c = b
        fc = fb
        d = b - a
        e = d  # previous step

        for _ in range(60):
            if not _check_time() or call_count >= MAX_CALLS - 2:
                break

            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b

            # Ensure fa and fb have opposite signs
            if fa * fb > 0:
                # Shouldn't happen if bracket is maintained, but safety
                break

            # Update c if it's no longer the best
            if abs(fc) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
                c = a
                fc = fa
                d = e = b - a

            # Set up IQUO
            m = (a + b) / 2.0
            e_prev = e
            e = d

            # Try inverse quadratic interpolation
            use_bisection = False
            try:
                # IQUO formula
                if a != c:
                    num1 = fa * fb
                    num2 = fc * fb
                    num3 = fc * fa
                    den = (fa - fb) * (fa - fc) * (fc - fb)
                    if den != 0:
                        s_iq = a * num1 / (num2 * num3) - b * num1 / (num2 * num3) \
                               - c * (num1 * (fa + fb + fc)) / (num2 * num3)
                        # Actually use the standard IQUO formula
                        s_iq = a * (fb * fc) / ((fa - fb) * (fa - fc)) \
                              - b * (fa * fc) / ((fb - fa) * (fb - fc)) \
                              + c * (fa * fb) / ((fc - fa) * (fc - fb))
                    else:
                        s_iq = None
                else:
                    s_iq = None

                if s_iq is not None and math.isfinite(s_iq):
                    # Check if IQUO is acceptable
                    # 1. s_iq must be within [a, b]
                    # 2. s_iq must not be too close to the midpoint
                    # 3. s_iq must not be too close to the last step
                    if s_iq <= a or s_iq >= b:
                        use_bisection = True
                    elif abs(s_iq - m) >= abs(e_prev) / 4.0:
                        # IQUO is too far from midpoint, use bisection
                        use_bisection = True
                    elif abs(s_iq - a) < abs(e_prev) / 2.0:
                        # IQUO is too close to a, use bisection
                        use_bisection = True
                else:
                    use_bisection = True
            except (ZeroDivisionError, OverflowError):
                use_bisection = True

            if use_bisection:
                s = m
                d = m - a
            else:
                s = s_iq
                d = s - a

            fs = _safe_f(s)
            if fs is None:
                break
            if abs(fs) <= TOL:
                return s

            # Update bracket
            if fa * fs < 0:
                b, fb = s, fs
            else:
                a, fa = s, fs

            # Update c (the point with smallest |f| among the three)
            if abs(fc) > abs(fb):
                c, fc = a, fa

        # Return the best point
        if abs(fa) < abs(fb):
            result_x = a
        else:
            result_x = b
        return result_x

    # Phase 3: No bracket found. Use secant/Newton hybrid.
    x_curr = x
    f_curr = fx
    x_prev = x + 1e-4
    f_prev = _safe_f(x_prev)
    if f_prev is None:
        return best_x

    for _ in range(30):
        if not _check_time() or call_count >= MAX_CALLS - 2:
            break
        if abs(f_curr) <= TOL:
            return x_curr
        if abs(f_prev) <= TOL:
            return x_prev

        if f_curr == f_prev:
            break

        # Secant step
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        if not math.isfinite(x_new):
            break

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        # Keep the two best points
        if abs(f_new) < abs(f_prev):
            x_prev, f_prev = x_curr, f_curr
        else:
            x_prev, f_prev = x_new, f_new
        x_curr, f_curr = x_new, f_new

    return best_x
# EVOLVE-BLOCK-END
