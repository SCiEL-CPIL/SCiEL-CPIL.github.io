"""Robust ill-conditioned root finder using inverse quadratic interpolation and model-and-optimize strategy."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    _t0 = time.time()
    _TIME_LIMIT = 135.0  # safety margin before 150s
    _MAX_EVALS = 490
    _evals = 0

    def _f(x):
        nonlocal _evals
        _evals += 1
        return f(x)

    def _time_ok():
        return (time.time() - _t0) < _TIME_LIMIT

    def _safe_f(x):
        try:
            val = float(_f(x))
            return val
        except Exception:
            return float('nan')

    # Best candidate tracking
    best_x = float(x0)
    best_fx = _safe_f(best_x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    def _update_best(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not math.isfinite(fx):
            return
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)

    _update_best(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    # --- Inverse quadratic interpolation step ---
    # Fits a parabola to (fx, x) points and finds where fx=0
    def _iqi_step(x1, fx1, x2, fx2, x3, fx3):
        # All three points must be distinct and finite
        if x1 == x2 or x2 == x3 or x1 == x3:
            return None
        if not all(math.isfinite(v) for v in [fx1, fx2, fx3]):
            return None
        
        # Inverse quadratic: x = a*fx^2 + b*fx + c
        # We want to find x where fx = 0
        # Using Lagrange interpolation in (f, x) space
        
        # If any fx is zero, we already have a root
        if fx1 == 0:
            return x1
        if fx2 == 0:
            return x2
        if fx3 == 0:
            return x3
        
        # Lagrange basis for f=0
        # L1(0) = (0 - fx2)(0 - fx3) / ((fx1 - fx2)(fx1 - fx3))
        # L2(0) = (0 - fx1)(0 - fx3) / ((fx2 - fx1)(fx2 - fx3))
        # L3(0) = (0 - fx1)(0 - fx2) / ((fx3 - fx1)(fx3 - fx2))
        
        d1 = (fx2 - fx1) * (fx3 - fx1)
        d2 = (fx1 - fx2) * (fx3 - fx2)
        d3 = (fx1 - fx3) * (fx2 - fx3)
        
        if d1 == 0 or d2 == 0 or d3 == 0:
            return None
        
        L1 = (fx2 * fx3) / d1
        L2 = (fx1 * fx3) / d2
        L3 = (fx1 * fx2) / d3
        
        x_new = L1 * x1 + L2 * x2 + L3 * x3
        
        if not math.isfinite(x_new):
            return None
        
        # Safeguard: limit step size
        ref_x = x3  # most recent point
        max_step = 5.0 * max(1.0, abs(ref_x))
        if abs(x_new - ref_x) > max_step:
            x_new = ref_x + max_step if x_new > ref_x else ref_x - max_step
        
        return x_new

    # --- Secant step (uses previous point, no extra derivative eval) ---
    def _secant_step(x_prev, fx_prev, x_curr, fx_curr):
        if x_prev == x_curr:
            return None
        d = (fx_curr - fx_prev) / (x_curr - x_prev)
        if d == 0 or not math.isfinite(d):
            return None
        step = fx_curr / d
        max_step = 5.0 * max(1.0, abs(x_curr))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        x_new = x_curr - step
        if not math.isfinite(x_new):
            return None
        return x_new

    # --- Bisection on bracket [a, b] where f(a)*f(b) <= 0 ---
    def _bisection(a, b, fa, fb, max_iter=100):
        if not _time_ok():
            return None
        # Ensure fa and fb have opposite signs or one is zero
        if not ((fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0)):
            return None
        for _ in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            mid = (a + b) / 2.0
            fmid = _safe_f(mid)
            if not math.isfinite(fmid):
                break
            _update_best(mid, fmid)
            if abs(fmid) <= 1e-8:
                return mid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
        mid = (a + b) / 2.0
        return mid

    # --- Robust IQUAD/Secant with bisection fallback ---
    def _iquad_with_bisection(x0_val, max_iter=60):
        x = float(x0_val)
        fx = _safe_f(x)
        if not math.isfinite(fx):
            return None
        _update_best(x, fx)
        if abs(fx) <= 1e-8:
            return x

        a, b = None, None
        fa, fb = None, None
        
        # Keep history of last 3 points for IQUAD
        history = [(x, fx)]  # list of (x, fx) tuples, most recent last

        for i in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break

            x_new = None
            
            # Try IQUAD first if we have 3 distinct points
            if len(history) >= 3:
                # Get last 3 distinct points
                h3 = history[-3:]
                x1, fx1 = h3[0]
                x2, fx2 = h3[1]
                x3, fx3 = h3[2]
                if x1 != x2 and x2 != x3 and x1 != x3:
                    x_new = _iqi_step(x1, fx1, x2, fx2, x3, fx3)
            
            # Fall back to secant if IQUAD fails or we don't have 3 points
            if x_new is None or not math.isfinite(x_new):
                if len(history) >= 2:
                    x_prev, fx_prev = history[-2]
                    x_curr, fx_curr = history[-1]
                    if x_prev != x_curr:
                        x_new = _secant_step(x_prev, fx_prev, x_curr, fx_curr)
            
            # If both fail, try bisection if bracket exists
            if x_new is None or not math.isfinite(x_new):
                if a is not None and b is not None and math.isfinite(fa) and math.isfinite(fb):
                    if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                        mid = (a + b) / 2.0
                        fmid = _safe_f(mid)
                        if not math.isfinite(fmid):
                            break
                        _update_best(mid, fmid)
                        if abs(fmid) <= 1e-8:
                            return mid
                        if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                            a, fa = mid, fmid
                        else:
                            b, fb = mid, fmid
                        if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                            break
                        history.append((mid, fmid))
                        if len(history) > 5:
                            history.pop(0)
                        x, fx = mid, fmid
                    else:
                        break
                else:
                    break
            else:
                fx_new = _safe_f(x_new)
                if not math.isfinite(fx_new):
                    # Method failed, try bisection if bracket exists
                    if a is not None and b is not None and math.isfinite(fa) and math.isfinite(fb):
                        if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                            mid = (a + b) / 2.0
                            fmid = _safe_f(mid)
                            if not math.isfinite(fmid):
                                break
                            _update_best(mid, fmid)
                            if abs(fmid) <= 1e-8:
                                return mid
                            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                                a, fa = mid, fmid
                            else:
                                b, fb = mid, fmid
                            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                                break
                            history.append((mid, fmid))
                            if len(history) > 5:
                                history.pop(0)
                            x, fx = mid, fmid
                        else:
                            break
                    else:
                        break
                else:
                    _update_best(x_new, fx_new)
                    if abs(fx_new) <= 1e-8:
                        return x_new

                    # Set bracket if sign change
                    if (fx_new <= 0 and fx >= 0) or (fx_new >= 0 and fx <= 0):
                        a, b = x, x_new
                        fa, fb = fx, fx_new
                        if abs(fa) > abs(fb):
                            a, b = b, a
                            fa, fb = fb, fa

                    # Update history
                    history.append((x_new, fx_new))
                    if len(history) > 5:
                        history.pop(0)
                    
                    x, fx = x_new, fx_new
                    if abs(x_new - x) < 1e-15 * max(1.0, abs(x)):
                        break

        return x if math.isfinite(x) else None

    # --- Phase 1: Try IQUAD from x0 ---
    result = _iquad_with_bisection(x0)
    if result is not None and math.isfinite(result):
        fr = _safe_f(result)
        if math.isfinite(fr) and abs(fr) <= 1e-8:
            return result
        if math.isfinite(fr):
            _update_best(result, fr)

    # --- Phase 2: Scan for sign changes around x0 ---
    base = float(x0)
    scan_scale = max(1.0, abs(base))

    # Try expanding brackets in multiple directions
    found_bracket = False
    for mult in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]:
        if not _time_ok():
            break
        if _evals >= _MAX_EVALS - 10:
            break
        for sign in [-1, 1]:
            a = base - sign * mult * scan_scale
            b = base + sign * mult * scan_scale
            if a == b:
                continue
            fa = _safe_f(a)
            fb = _safe_f(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            _update_best(a, fa)
            _update_best(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                # Found bracket, apply bisection
                res = _bisection(a, b, fa, fb, max_iter=80)
                if res is not None and math.isfinite(res):
                    fres = _safe_f(res)
                    if math.isfinite(fres) and abs(fres) <= 1e-8:
                        return res
                    _update_best(res, fres)
                found_bracket = True
                break
        if found_bracket:
            break

    # --- Phase 3: Local search for sign changes near best_x ---
    if not _time_ok() or _evals >= _MAX_EVALS - 10:
        return best_x

    center = best_x
    scale = max(1.0, abs(center))

    # Fine-grained scan near best_x
    for mult in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        if not _time_ok():
            break
        if _evals >= _MAX_EVALS - 10:
            break
        for sign in [-1, 1]:
            a = center - sign * mult * scale
            b = center + sign * mult * scale
            if a == b:
                continue
            fa = _safe_f(a)
            fb = _safe_f(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            _update_best(a, fa)
            _update_best(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                res = _bisection(a, b, fa, fb, max_iter=60)
                if res is not None and math.isfinite(res):
                    fres = _safe_f(res)
                    if math.isfinite(fres) and abs(fres) <= 1e-8:
                        return res
                    _update_best(res, fres)
                break

    # --- Phase 4: Golden-section search on |f| as fallback ---
    if not _time_ok() or _evals >= _MAX_EVALS - 10:
        return best_x

    center = best_x
    scale = max(1.0, abs(center))
    a = center - 5.0 * scale
    b = center + 5.0 * scale

    # Ensure finite values at endpoints
    for mult in [5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
        a = center - mult * scale
        b = center + mult * scale
        ga = _safe_f(a)
        gb = _safe_f(b)
        if math.isfinite(ga) and math.isfinite(gb):
            break

    if math.isfinite(ga) and math.isfinite(gb):
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _safe_f(c)
        fd = _safe_f(d)
        _update_best(c, fc)
        _update_best(d, fd)

        for _ in range(40):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            if abs(fc) <= 1e-8:
                return c
            if abs(fd) <= 1e-8:
                return d
            if abs(fc) < abs(fd):
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = _safe_f(c)
                _update_best(c, fc)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = _safe_f(d)
                _update_best(d, fd)
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        mid = (a + b) / 2.0
        fmid = _safe_f(mid)
        _update_best(mid, fmid)
        if abs(fmid) <= 1e-8:
            return mid

    # --- Phase 5: Final IQUAD/Secant polish from best_x with tighter tolerance ---
    if _time_ok() and _evals < _MAX_EVALS - 10:
        x = best_x
        history = [(x, _safe_f(x))]
        
        for _ in range(50):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            fx = history[-1][1]
            if not math.isfinite(fx):
                break
            _update_best(history[-1][0], fx)
            if abs(fx) <= 1e-10:  # Tighter tolerance for polish
                return history[-1][0]
            
            x_new = None
            # Try IQUAD
            if len(history) >= 3:
                h3 = history[-3:]
                x1, fx1 = h3[0]
                x2, fx2 = h3[1]
                x3, fx3 = h3[2]
                if x1 != x2 and x2 != x3 and x1 != x3:
                    x_new = _iqi_step(x1, fx1, x2, fx2, x3, fx3)
            
            # Fall back to secant
            if x_new is None or not math.isfinite(x_new):
                if len(history) >= 2:
                    x_prev, fx_prev = history[-2]
                    x_curr, fx_curr = history[-1]
                    if x_prev != x_curr:
                        x_new = _secant_step(x_prev, fx_prev, x_curr, fx_curr)
            
            if x_new is None or not math.isfinite(x_new):
                break
            
            fx_new = _safe_f(x_new)
            if not math.isfinite(fx_new):
                break
            
            # Check convergence
            if abs(x_new - history[-1][0]) < 1e-14 * max(1.0, abs(x_new)):
                break
            
            history.append((x_new, fx_new))
            if len(history) > 5:
                history.pop(0)
        
        # Final check
        if history:
            x_final, fx_final = history[-1]
            if math.isfinite(fx_final) and abs(fx_final) <= 1e-8:
                return x_final
            _update_best(x_final, fx_final)

    return best_x
# EVOLVE-BLOCK-END
