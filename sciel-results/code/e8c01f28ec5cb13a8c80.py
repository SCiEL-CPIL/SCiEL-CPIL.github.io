"""Root finding for ill-conditioned equations: model-and-optimize approach."""
import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    Returns the best x found (smallest |f(x)|) within the budget.
    """
    start_time = time.time()
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    deadline = 145.0

    best_x = float(x0)
    try:
        best_fx = float(f(best_x))
        eval_count += 1
        best_abs_fx = abs(best_fx) if math.isfinite(best_fx) else float('inf')
    except Exception:
        best_fx = 0.0
        best_abs_fx = float('inf')

    def _time_left():
        return deadline - (time.time() - start_time)

    def _eval(x):
        """Evaluate f(x) with eval-count and time guards. Returns (fx, ok)."""
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if eval_count >= max_evals or _time_left() <= 0.5:
            return None, False
        try:
            fx = float(f(x))
        except Exception:
            return None, False
        eval_count += 1
        if not math.isfinite(fx):
            return None, False
        abs_fx = abs(fx)
        if abs_fx < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs_fx
        return fx, True

    def _check_done():
        return best_abs_fx <= tol

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 1: Adaptive secant with model-based step control
    # ------------------------------------------------------------------
    x1 = float(x0)
    f1 = best_fx
    h_init = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h_init
    f2, ok = _eval(x2)
    if not ok:
        x2 = x1 - h_init
        f2, ok = _eval(x2)
    if not ok:
        return best_x

    # Track model quality: if secant steps are making progress, keep going
    prev_abs_f = abs(f2)
    consecutive_bad = 0
    max_consecutive_bad = 10

    for it in range(200):
        if _check_done():
            return best_x
        if eval_count >= max_evals or _time_left() < 2.0:
            break

        dx = x2 - x1
        if abs(dx) < 1e-15:
            break
        d = (f2 - f1) / dx
        if abs(d) < 1e-15:
            break

        # Model-based: compute secant step
        x_new = x2 - f2 / d

        # Adaptive safeguard: limit step based on how well the model fits
        # If we're making good progress, allow larger steps
        max_step = 10.0 * max(1.0, abs(x2))
        if abs(x_new - x2) > max_step:
            x_new = x2 - math.copysign(max_step, x_new - x2)

        f_new, ok = _eval(x_new)
        if not ok:
            break

        if abs(f_new) <= tol:
            return x_new

        # Track progress
        if abs(f_new) < abs(f2) * 0.95:
            consecutive_bad = 0
        else:
            consecutive_bad += 1

        # If too many bad steps in a row, break out
        if consecutive_bad >= max_consecutive_bad:
            break

        x1, f1 = x2, f2
        x2, f2 = x_new, f_new

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 2: Bracket search + bisection with inverse quadratic interpolation
    # ------------------------------------------------------------------
    a = x2
    fa = f2
    if fa is None:
        fa, _ = _eval(a)
        if fa is None:
            return best_x

    # Expand bracket outward with multiple strategies
    b = None
    fb = None
    scale = 1.0
    for _ in range(30):
        if _time_left() < 2.0 or eval_count >= max_evals - 15:
            break
        b_pos = a + scale * max(1.0, abs(a))
        fb_pos, ok1 = _eval(b_pos)
        if ok1 and fa * fb_pos < 0:
            b = b_pos
            fb = fb_pos
            break
        b_neg = a - scale * max(1.0, abs(a))
        fb_neg, ok2 = _eval(b_neg)
        if ok2 and fa * fb_neg < 0:
            b = b_neg
            fb = fb_neg
            break
        scale *= 1.5

    # If no bracket found, try from x0 with different scales
    if b is None:
        a2 = float(x0)
        fa2, ok = _eval(a2)
        if ok:
            scale2 = 0.1
            for _ in range(25):
                if _time_left() < 2.0 or eval_count >= max_evals - 15:
                    break
                b_pos2 = a2 + scale2 * max(1.0, abs(a2))
                fb_pos2, ok1 = _eval(b_pos2)
                if ok1 and fa2 * fb_pos2 < 0:
                    b = b_pos2
                    fb = fb_pos2
                    a = a2
                    fa = fa2
                    break
                b_neg2 = a2 - scale2 * max(1.0, abs(a2))
                fb_neg2, ok2 = _eval(b_neg2)
                if ok2 and fa2 * fb_neg2 < 0:
                    b = b_neg2
                    fb = fb_neg2
                    a = a2
                    fa = fa2
                    break
                scale2 *= 2.0

    if b is not None and fb is not None and fa * fb < 0:
        lo, hi = min(a, b), max(a, b)
        flo = fa if a == lo else fb
        fhi = fa if a == hi else fb
        if flo * fhi > 0:
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        for _ in range(80):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break

            if abs(flo) <= tol:
                return lo
            if abs(fhi) <= tol:
                return hi

            # Inverse quadratic interpolation
            s = None
            denom = fhi * (flo + fhi) + flo * (fhi - flo)
            if denom != 0:
                s = lo + (hi - lo) * flo * (flo - fhi) / denom
                if not math.isfinite(s) or s <= lo or s >= hi:
                    s = None

            if s is None:
                s = 0.5 * (lo + hi)

            fs, ok = _eval(s)
            if not ok:
                break

            if abs(fs) <= tol:
                return s

            if flo * fs < 0:
                hi, fhi = s, fs
            else:
                lo, flo = s, fs

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 3: Multi-start secant from diverse points
    # ------------------------------------------------------------------
    offsets = [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0,
               0.01, -0.01, 0.5, -0.5, 5.0, -5.0]
    for off in offsets:
        if eval_count >= max_evals - 20 or _time_left() < 2.0:
            break
        x_try = float(x0) + off * max(1.0, abs(x0))
        fx_try, ok = _eval(x_try)
        if not ok:
            continue
        if abs(fx_try) <= tol:
            return x_try

        # Quick secant from this point
        h_r = 1e-4 * max(1.0, abs(x_try))
        x_r2 = x_try + h_r
        fx_r2, ok2 = _eval(x_r2)
        if not ok2:
            x_r2 = x_try - h_r
            fx_r2, ok2 = _eval(x_r2)
        if not ok2:
            continue

        consecutive_bad_r = 0
        for _ in range(15):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break
            dx_r = x_r2 - x_try
            if abs(dx_r) < 1e-15:
                break
            d_r = (fx_r2 - fx_try) / dx_r
            if abs(d_r) < 1e-15:
                break
            x_new = x_r2 - fx_r2 / d_r
            fx_new, ok3 = _eval(x_new)
            if not ok3:
                break
            if abs(fx_new) <= tol:
                return x_new
            if abs(fx_new) < abs(fx_r2) * 0.95:
                consecutive_bad_r = 0
            else:
                consecutive_bad_r += 1
                if consecutive_bad_r >= 4:
                    break
            x_try, fx_try = x_r2, fx_r2
            x_r2, fx_r2 = x_new, fx_new

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 4: High-precision polish pass with tighter tolerance
    # ------------------------------------------------------------------
    if eval_count < max_evals - 20 and _time_left() > 3.0:
        if abs(best_fx) < 1e-3:
            x_polish = best_x
            fx_polish = best_fx
            h_p = 1e-10 * max(1.0, abs(x_polish))
            xp2 = x_polish + h_p
            fp2, ok = _eval(xp2)
            if not ok:
                xp2 = x_polish - h_p
                fp2, ok = _eval(xp2)
            if ok:
                # Use tighter tolerance for polish: aim for 1e-10
                tol_polish = min(tol, 1e-10)
                for _ in range(80):
                    if _check_done():
                        return best_x
                    if eval_count >= max_evals or _time_left() < 1.0:
                        break
                    dx_p = xp2 - x_polish
                    if abs(dx_p) < 1e-18:
                        break
                    d_p = (fp2 - fx_polish) / dx_p
                    if abs(d_p) < 1e-18:
                        break
                    x_new = xp2 - fp2 / d_p
                    # Safeguard: don't step too far
                    max_step_p = 100.0 * max(1e-15, abs(xp2))
                    if abs(x_new - xp2) > max_step_p:
                        x_new = xp2 - math.copysign(max_step_p, x_new - xp2)
                    fx_new, ok = _eval(x_new)
                    if not ok:
                        break
                    if abs(fx_new) <= tol_polish:
                        return x_new
                    if abs(fx_new) < abs(fp2) * 0.999:
                        x_polish, fx_polish = xp2, fp2
                        xp2, fp2 = x_new, fx_new
                    else:
                        # Try bisection-like fallback for polish
                        if abs(fx_new) < abs(fx_polish):
                            # Keep x_new as new best
                            x_polish, fx_polish = xp2, fp2
                            xp2, fp2 = x_new, fx_new
                        else:
                            break

    return best_x
