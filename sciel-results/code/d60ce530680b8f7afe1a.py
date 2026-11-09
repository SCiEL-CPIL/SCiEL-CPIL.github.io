"""Direct root-finder: secant-primary with IQUIN polish, bracket-bisect fallback."""
import time
import math

def find_root(f, x0):
    _t0 = time.time()
    _DEADLINE = 145.0
    _MAX_EVALS = 490
    _evals = 0

    def _f(x):
        nonlocal _evals
        _evals += 1
        try:
            val = f(x)
            return float(val)
        except Exception:
            return float('nan')

    def _time_ok():
        return (time.time() - _t0) < _DEADLINE

    def _budget_ok():
        return _evals < _MAX_EVALS and _time_ok()

    best_x = float(x0)
    best_fx = _f(best_x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    def _update(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not math.isfinite(fx):
            return
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)

    _update(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    def _bisection(a, b, fa, fb):
        nonlocal best_x, best_fx, best_abs
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        for _ in range(200):
            if not _budget_ok():
                break
            mid = (a + b) / 2.0
            if mid == a or mid == b:
                break
            fmid = _f(mid)
            _update(mid, fmid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= 1e-8:
                return mid, fmid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
        return mid, fmid

    def _secant_step(x, fx, x_prev, fx_prev):
        if fx_prev == fx:
            return None
        x_new = x - fx * (x - x_prev) / (fx - fx_prev)
        if not math.isfinite(x_new):
            return None
        return x_new

    def _iquin_step(x0_, f0_, x1_, f1_, x2_, f2_):
        """Inverse quadratic interpolation: fit parabola through (f_i, x_i).
        Returns candidate x or None."""
        # Need distinct f values
        if f0_ == f1_ or f1_ == f2_ or f0_ == f2_:
            return None
        # Lagrange interpolation: x = sum f_i * L_i(f) where L_i are Lagrange basis
        # L_0(f) = (f - f1)(f - f2) / ((f0-f1)(f0-f2))
        # L_1(f) = (f - f0)(f - f2) / ((f1-f0)(f1-f2))
        # L_2(f) = (f - f0)(f - f1) / ((f2-f0)(f2-f1))
        # We want x such that f(x) = 0, so evaluate at f=0:
        # L_0(0) = (0-f1)(0-f2) / ((f0-f1)(f0-f2)) = f1*f2 / ((f0-f1)(f0-f2))
        # L_1(0) = f0*f2 / ((f1-f0)(f1-f2))
        # L_2(0) = f0*f1 / ((f2-f0)(f2-f1))
        d0 = (f0_ - f1_) * (f0_ - f2_)
        d1 = (f1_ - f0_) * (f1_ - f2_)
        d2 = (f2_ - f0_) * (f2_ - f1_)
        if abs(d0) < 1e-30 or abs(d1) < 1e-30 or abs(d2) < 1e-30:
            return None
        x_new = x0_ * (f1_ * f2_) / d0 + x1_ * (f0_ * f2_) / d1 + x2_ * (f0_ * f1_) / d2
        if not math.isfinite(x_new):
            return None
        return x_new

    # --- Phase 1: Secant-primary with bracket tracking ---
    x = float(x0)
    fx = _f(x)
    _update(x, fx)
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x

    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None
    x_prev = x + 1e-6 * max(1.0, abs(x))
    fx_prev = _f(x_prev)
    _update(x_prev, fx_prev)

    # Keep history for IQUIN: (x, f) pairs
    hist_x = [x, x_prev]
    hist_f = [fx, fx_prev]

    for i in range(100):
        if not _budget_ok():
            break
        if abs(fx) <= 1e-8:
            return x

        # Try IQUIN if we have 3 distinct points
        x_new = None
        if len(hist_x) >= 3:
            x_new = _iquin_step(hist_x[-3], hist_f[-3], hist_x[-2], hist_f[-2], hist_x[-1], hist_f[-1])
        
        # Fallback to secant
        if x_new is None:
            x_new = _secant_step(x, fx, x_prev, fx_prev)
        
        if x_new is None:
            break

        fx_new = _f(x_new)
        _update(x_new, fx_new)
        if not math.isfinite(fx_new):
            break
        if abs(fx_new) <= 1e-8:
            return x_new

        # Track bracket on sign change
        if (fx_new <= 0 and fx >= 0) or (fx_new >= 0 and fx <= 0):
            if abs(fx) <= abs(fx_new):
                bracket_a, bracket_b = x, x_new
                bracket_fa, bracket_fb = fx, fx_new
            else:
                bracket_a, bracket_b = x_new, x
                bracket_fb, bracket_fa = fx, fx_new

        # Update history
        hist_x.append(x_new)
        hist_f.append(fx_new)
        if len(hist_x) > 5:
            hist_x = hist_x[-5:]
            hist_f = hist_f[-5:]

        x_prev, fx_prev = x, fx
        x, fx = x_new, fx_new
        if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
            break

    if abs(fx) <= 1e-8 and math.isfinite(fx):
        return x
    _update(x, fx)

    # If we have a bracket, bisect
    if bracket_a is not None and bracket_b is not None:
        if (bracket_fa <= 0 and bracket_fb >= 0) or (bracket_fa >= 0 and bracket_fb <= 0):
            mid, fmid = _bisection(bracket_a, bracket_b, bracket_fa, bracket_fb)
            _update(mid, fmid)
            if abs(fmid) <= 1e-8:
                return mid

    # --- Phase 2: Scan for a bracket around x0 ---
    if _budget_ok() and best_abs > 1e-8:
        base = float(x0)
        scan_scale = max(1.0, abs(base))
        for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]:
            if not _budget_ok():
                break
            for sign in [-1, 1]:
                if not _budget_ok():
                    break
                a = base - sign * mult * scan_scale
                b = base + sign * mult * scan_scale
                if a == b:
                    continue
                fa = _f(a)
                fb = _f(b)
                _update(a, fa)
                _update(b, fb)
                if not (math.isfinite(fa) and math.isfinite(fb)):
                    continue
                if abs(fa) <= 1e-8:
                    return a
                if abs(fb) <= 1e-8:
                    return b
                if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                    mid, fmid = _bisection(a, b, fa, fb)
                    _update(mid, fmid)
                    if abs(fmid) <= 1e-8:
                        return mid
                    break
            else:
                continue
            break

    # --- Phase 3: IQUIN polish from best_x ---
    if _budget_ok() and best_abs > 1e-8:
        x = best_x
        fx = _f(x)
        _update(x, fx)
        if not math.isfinite(fx):
            return best_x

        # Build initial history with secant-like perturbation
        h = 1e-6 * max(1.0, abs(x))
        x1 = x + h
        fx1 = _f(x1)
        _update(x1, fx1)
        if not math.isfinite(fx1):
            return best_x

        # Get a third point for IQUIN
        x2 = x - h
        fx2 = _f(x2)
        _update(x2, fx2)
        if not math.isfinite(fx2):
            return best_x

        hx = [x, x1, x2]
        hf = [fx, fx1, fx2]

        for _ in range(50):
            if not _budget_ok():
                break
            if abs(fx) <= 1e-8:
                return x

            x_new = _iquin_step(hx[0], hf[0], hx[1], hf[1], hx[2], hf[2])
            if x_new is None:
                # Fallback to secant using last two
                x_new = _secant_step(hx[2], hf[2], hx[1], hf[1])
                if x_new is None:
                    break

            fx_new = _f(x_new)
            _update(x_new, fx_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) <= 1e-8:
                return x_new

            # Only accept if it improves
            if abs(fx_new) < abs(fx):
                hx = [hx[1], hx[2], x_new]
                hf = [hf[1], hf[2], fx_new]
                x, fx = x_new, fx_new
            else:
                # Replace worst point
                idx_max = 0
                for i in range(1, 3):
                    if abs(hf[i]) > abs(hf[idx_max]):
                        idx_max = i
                hx[idx_max] = x_new
                hf[idx_max] = fx_new

            if abs(x_new - hx[2]) < 1e-15 * max(1.0, abs(x_new)):
                break

    return best_x
