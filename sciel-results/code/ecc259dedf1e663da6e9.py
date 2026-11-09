"""Derivative-free inverse quadratic interpolation (IQI) root finder with bracketing."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: return the best point found before wall-clock limit.
    t_start = time.time()
    deadline = 140.0  # 120s budget, 20s safety margin

    def _elapsed():
        return time.time() - t_start

    def _check_deadline(best_x, best_fx):
        if _elapsed() > deadline:
            return (best_x, best_fx, True)
        return (None, None, False)

    x = float(x0)
    try:
        fx = float(f(x))
    except Exception:
        # If even the initial eval fails, return x0 as a last resort.
        return x

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-12:
        return x

    # Phase 1: Acquire a bracket [a, b] with f(a)*f(b) <= 0.
    a = x
    fa = fx
    # Try expanding in both directions to find a sign change.
    h = 1.0
    b = None
    fb = None

    # First, try a small step in each direction.
    for step in [1.0, -1.0]:
        xb = x + step * h
        try:
            fb = float(f(xb))
        except Exception:
            continue
        if math.isfinite(fb) and fa * fb <= 0:
            a = min(x, xb)
            b = max(x, xb)
            fa = f(a) if a != x else fx
            fb = f(b) if b != x else fx
            break

    # If no bracket found with h=1, try expanding.
    if b is None:
        for h in [0.5, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
            for step in [1.0, -1.0]:
                xb = x + step * h
                try:
                    fb = float(f(xb))
                except Exception:
                    continue
                if math.isfinite(fb) and fa * fb <= 0:
                    a = min(x, xb)
                    b = max(x, xb)
                    fa = f(a) if a != x else fx
                    fb = f(b) if b != x else fx
                    break
            if b is not None:
                break

    # If still no bracket, fall back to a pure bisection-free approach:
    # just use IQI without a bracket (risky but may work for polynomials).
    # For robustness, if no bracket, do a local search.
    if b is None:
        # No bracket: use a secant-like approach with IQI, no safeguarding.
        # This is a fallback; polynomials often have nearby roots.
        x_prev = x
        fx_prev = fx
        x_curr = x + 1e-4
        try:
            fx_curr = float(f(x_curr))
        except Exception:
            return x

        # IQI loop without bracket
        for _ in range(50):
            if _elapsed() > deadline:
                break
            if abs(fx_curr) <= 1e-12:
                return x_curr
            if abs(fx_prev) <= abs(fx_curr):
                return x_prev

            # Need a third point for IQI; use a small perturbation.
            x_third = x_curr + 1e-6
            try:
                fx_third = float(f(x_third))
            except Exception:
                # Fall back to secant
                if fx_prev != fx_curr:
                    x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                else:
                    break
                try:
                    fx_new = float(f(x_new))
                except Exception:
                    break
                if not math.isfinite(fx_new):
                    break
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = x_new, fx_new
                continue

            # IQI with points (x_prev, fx_prev), (x_curr, fx_curr), (x_third, fx_third)
            x_new = _iqi_step(x_prev, fx_prev, x_curr, fx_curr, x_third, fx_third)
            if x_new is None or not math.isfinite(x_new):
                break

            try:
                fx_new = float(f(x_new))
            except Exception:
                break
            if not math.isfinite(fx_new):
                break

            # Update the three points: keep the two closest to root.
            if abs(fx_new) <= abs(fx_prev):
                x_prev, fx_prev = x_third, fx_third
            elif abs(fx_new) <= abs(fx_curr):
                x_prev, fx_prev = x_curr, fx_curr
            else:
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = x_third, fx_third
                continue

            x_curr, fx_curr = x_new, fx_new

        # Return the best
        if abs(fx_prev) <= abs(fx_curr):
            return x_prev
        return x_curr

    # Phase 2: IQI with bracketing.
    # We have [a, b] with fa*fb <= 0.
    # Evaluate the midpoint to get a third point for IQI.
    mid = 0.5 * (a + b)
    try:
        fmid = float(f(mid))
    except Exception:
        return a if abs(fa) < abs(fb) else b

    if abs(fmid) <= 1e-12:
        return mid

    # Points: (a, fa), (b, fb), (mid, fmid)
    p1, f1 = a, fa
    p2, f2 = b, fb
    p3, f3 = mid, fmid

    # Keep the best point (smallest |f|) as a candidate.
    best_x = x
    best_fx = fx

    for _ in range(80):
        # Deadline check
        if _elapsed() > deadline:
            break

        # Check if any point is a root
        if abs(f1) <= 1e-12:
            return p1
        if abs(f2) <= 1e-12:
            return p2
        if abs(f3) <= 1e-12:
            return p3

        # Update best
        if abs(f1) < abs(best_fx):
            best_x, best_fx = p1, f1
        if abs(f2) < abs(best_fx):
            best_x, best_fx = p2, f2
        if abs(f3) < abs(best_fx):
            best_x, best_fx = p3, f3

        # Determine the bracket: the two points with opposite signs.
        # Find which pair brackets the root.
        bracket = None
        if f1 * f2 <= 0:
            bracket = (p1, f1, p2, f2)
        elif f1 * f3 <= 0:
            bracket = (p1, f1, p3, f3)
        elif f2 * f3 <= 0:
            bracket = (p2, f2, p3, f3)

        if bracket is None:
            # No bracket: all same sign. This shouldn't happen if we started
            # with a bracket, but handle gracefully.
            # Fall back to bisection on the widest range.
            lo = min(p1, p2, p3)
            hi = max(p1, p2, p3)
            x_new = 0.5 * (lo + hi)
            try:
                f_new = float(f(x_new))
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            # Replace the point with the largest |f|
            if abs(f1) >= abs(f2) and abs(f1) >= abs(f3):
                p1, f1 = x_new, f_new
            elif abs(f2) >= abs(f3):
                p2, f2 = x_new, f_new
            else:
                p3, f3 = x_new, f_new
            continue

        la, fa_b, lb, fb_b = bracket
        # The third point is the one not in the bracket.
        if bracket == (p1, f1, p2, f2):
            p_third, f_third = p3, f3
        elif bracket == (p1, f1, p3, f3):
            p_third, f_third = p2, f2
        else:
            p_third, f_third = p1, f1

        # IQI step: find root of quadratic through (la, fa_b), (lb, fb_b), (p_third, f_third)
        x_iqi = _iqi_step(la, fa_b, lb, fb_b, p_third, f_third)

        # Safeguard: if IQI step is outside the bracket or too close to existing points,
        # use bisection.
        x_bisect = 0.5 * (la + lb)
        use_iqi = False
        if x_iqi is not None and math.isfinite(x_iqi):
            if la < x_iqi < lb:  # strictly inside bracket
                # Check not too close to existing points
                min_dist = min(abs(x_iqi - la), abs(x_iqi - lb), abs(x_iqi - p_third))
                bracket_width = abs(lb - la)
                if min_dist > 1e-14 * max(1.0, abs(x_iqi)) and min_dist > 0.1 * bracket_width:
                    use_iqi = True

        if use_iqi:
            x_new = x_iqi
        else:
            x_new = x_bisect

        try:
            f_new = float(f(x_new))
        except Exception:
            break
        if not math.isfinite(f_new):
            break

        if abs(f_new) <= 1e-12:
            return x_new

        # Update points: replace the point in the bracket that is NOT the new point's
        # sign partner. The new point is in [la, lb]. The new bracket will be
        # [la, x_new] or [x_new, lb] depending on sign.
        # The point to replace is the one in the bracket whose sign matches f_new
        # (since the new point and the opposite-sign endpoint form the new bracket).
        if fa_b * f_new <= 0:
            # New bracket is [la, x_new], so replace lb
            if lb == p2:
                p2, f2 = x_new, f_new
            elif lb == p3:
                p3, f3 = x_new, f_new
            else:
                p1, f1 = x_new, f_new
        else:
            # New bracket is [x_new, lb], so replace la
            if la == p1:
                p1, f1 = x_new, f_new
            elif la == p3:
                p3, f3 = x_new, f_new
            else:
                p2, f2 = x_new, f_new

        # Check bracket width for convergence
        # Find current bracket
        if f1 * f2 <= 0:
            bw = abs(p2 - p1)
        elif f1 * f3 <= 0:
            bw = abs(p3 - p1)
        elif f2 * f3 <= 0:
            bw = abs(p3 - p2)
        else:
            bw = float('inf')

        if bw < 1e-14 * max(1.0, abs(best_x)):
            break

    # Return the best point found
    if abs(f1) < abs(best_fx):
        best_x = p1
    if abs(f2) < abs(best_fx):
        best_x = p2
    if abs(f3) < abs(best_fx):
        best_x = p3

    return best_x


def _iqi_step(x1, f1, x2, f2, x3, f3):
    """
    Inverse quadratic interpolation: find the root of the quadratic
    that passes through (x1, f1), (x2, f2), (x3, f3) in the (f, x) plane,
    i.e., fit x as a quadratic function of f and find where x(f=0).
    """
    # Lagrange interpolation for x as a function of f.
    # x(f) = f1^2 * f2^2 / ((f1-f2)(f1-f3)) * x1 + ... (inverse quadratic)
    # Standard formula:
    # x = f^2 * [ x1 / ((f1-f2)(f1-f3)) + x2 / ((f2-f1)(f2-f3)) + x3 / ((f3-f1)(f3-f2)) ]
    # evaluated at f=0:
    # x(0) = 0^2 * [...] = 0? No, that's wrong.
    
    # Actually, inverse quadratic interpolation fits x = c0 + c1*f + c2*f^2.
    # Solving for c0, c1, c2 from the three points:
    # c0 + c1*f1 + c2*f1^2 = x1
    # c0 + c1*f2 + c2*f2^2 = x2
    # c0 + c1*f3 + c2*f3^2 = x3
    # We want x(0) = c0.
    
    # Using Lagrange:
    # c0 = x1 * L1(0) + x2 * L2(0) + x3 * L3(0)
    # where L1(f) = (f - f2)(f - f3) / ((f1 - f2)(f1 - f3))
    # L1(0) = (0 - f2)(0 - f3) / ((f1 - f2)(f1 - f3)) = f2*f3 / ((f1-f2)(f1-f3))
    
    denom1 = (f1 - f2) * (f1 - f3)
    denom2 = (f2 - f1) * (f2 - f3)
    denom3 = (f3 - f1) * (f3 - f2)
    
    if abs(denom1) < 1e-300 or abs(denom2) < 1e-300 or abs(denom3) < 1e-300:
        return None
    
    c0 = x1 * (f2 * f3) / denom1 + x2 * (f1 * f3) / denom2 + x3 * (f1 * f2) / denom3
    
    if not math.isfinite(c0):
        return None
    
    return c0

# EVOLVE-BLOCK-END
