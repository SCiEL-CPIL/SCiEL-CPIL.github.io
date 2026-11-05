import math
import time
import random


def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 138.0  # safety margin before the 150s kill

    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = math.inf

    # Seeded RNG for reproducible multi-start
    rng = random.Random(42)

    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() >= DEADLINE:
            return None
        evals += 1
        try:
            val = float(f(x))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError,
                AttributeError, ArithmeticError):
            return None
        if not math.isfinite(val):
            return None
        av = abs(val)
        if av < best_abs:
            best_abs = av
            best_x = float(x)
            best_f = val
        return val

    def is_good(x, fx):
        return fx is not None and abs(fx) <= TOL

    def is_close(x, fx):
        """Check if we're close enough to start a polish pass."""
        return fx is not None and abs(fx) <= 1e-5

    def safe_get_deriv():
        try:
            d = f.deriv
            if callable(d):
                return d
        except Exception:
            pass
        try:
            d = f.derivative
            if callable(d):
                return d
        except Exception:
            pass
        return None

    def siqi(x0, x1, x2):
        """Safeguarded Inverse Quadratic Interpolation (SIQI).
        Uses three points and falls back to bisection if the quadratic step
        would leave the bracket or be too small.
        """
        a = float(x0)
        b = float(x1)
        c = float(x2)
        fa = call_f(a)
        fb = call_f(b)
        fc = call_f(c)

        if fa is None and fb is None and fc is None:
            return a, None

        # Pick the three finite points
        pts = []
        for xv, fxv in [(a, fa), (b, fb), (c, fc)]:
            if fxv is not None:
                pts.append((xv, fxv))
        if len(pts) < 2:
            if pts:
                return pts[0][0], pts[0][1]
            return a, None
        if len(pts) == 2:
            # Fall back to secant-like behavior
            (x1, f1), (x2, f2) = pts
            if is_good(x1, f1):
                return x1, f1
            if is_good(x2, f2):
                return x2, f2
            return secant(x1, x2)

        (x1, f1), (x2, f2), (x3, f3) = pts

        if is_good(x1, f1):
            return x1, f1
        if is_good(x2, f2):
            return x2, f2
        if is_good(x3, f3):
            return x3, f3

        for _ in range(80):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break

            # Find a bracketing interval among the three points
            # Sort by x value
            pts_sorted = sorted([(x1, f1), (x2, f2), (x3, f3)], key=lambda p: p[0])
            (xa, fa_), (xb, fb_), (xc, fc_) = pts_sorted

            # Find sign change
            bracket = None
            if fa_ * fb_ < 0:
                bracket = (xa, fa_, xb, fb_)
            elif fb_ * fc_ < 0:
                bracket = (xb, fb_, xc, fc_)
            elif fa_ * fc_ < 0:
                bracket = (xa, fa_, xc, fc_)

            if bracket is None:
                # No sign change; use inverse quadratic interpolation on all three
                # IQR: x_new = x1*f2*f3/((f2-f1)(f3-f1)) + x2*f1*f3/((f1-f2)(f3-f2)) + x3*f1*f2/((f1-f3)(f2-f3))
                denom1 = (f2 - f1) * (f3 - f1)
                denom2 = (f1 - f2) * (f3 - f2)
                denom3 = (f1 - f3) * (f2 - f3)
                xn = None
                if abs(denom1) > 1e-300 and abs(denom2) > 1e-300 and abs(denom3) > 1e-300:
                    xn = (x1 * f2 * f3 / denom1 + x2 * f1 * f3 / denom2 + x3 * f1 * f2 / denom3)
                if xn is None or not math.isfinite(xn):
                    break

                # Check if xn is within the range of the three points
                x_min = min(x1, x2, x3)
                x_max = max(x1, x2, x3)
                margin = 0.5 * (x_max - x_min)
                if xn < x_min - margin or xn > x_max + margin:
                    break

                fxn = call_f(xn)
                if fxn is None:
                    break
                if is_good(xn, fxn):
                    return xn, fxn

                # Replace the point with the largest |f|
                if abs(fxn) < abs(f1):
                    x1, f1 = xn, fxn
                elif abs(fxn) < abs(f2):
                    x2, f2 = xn, fxn
                else:
                    x3, f3 = xn, fxn
                continue

            # Have a bracket [bl, br] with sign change
            bl, fbl, br, fbr = bracket
            if is_good(bl, fbl):
                return bl, fbl
            if is_good(br, fbr):
                return br, fbr

            # Inverse quadratic interpolation using the two bracket endpoints and the third point
            # Use the two bracket points and the midpoint-like third
            # IQR formula with three points
            # Pick the third point as the one not in the bracket
            if (bl, fbl) == (x1, f1) or (br, fbr) == (x1, f1):
                third = (x2, f2) if not ((bl, fbl) == (x2, f2) or (br, fbr) == (x2, f2)) else (x3, f3)
            elif (bl, fbl) == (x2, f2) or (br, fbr) == (x2, f2):
                third = (x1, f1) if not ((bl, fbl) == (x1, f1) or (br, fbr) == (x1, f1)) else (x3, f3)
            else:
                third = (x1, f1) if not ((bl, fbl) == (x1, f1) or (br, fbr) == (x1, f1)) else (x2, f2)

            x3pt, f3pt = third

            # IQR: use bl, br, and x3pt
            denom1 = (fbr - fbl) * (f3pt - fbl)
            denom2 = (fbl - fbr) * (f3pt - fbr)
            denom3 = (fbl - f3pt) * (fbr - f3pt)

            xn = None
            if abs(denom1) > 1e-300 and abs(denom2) > 1e-300 and abs(denom3) > 1e-300:
                xn = (bl * fbr * f3pt / denom1 + br * fbl * f3pt / denom2 + x3pt * fbl * fbr / denom3)

            if xn is None or not math.isfinite(xn):
                # Fall back to bisection
                xn = 0.5 * (bl + br)

            # Safeguard: keep xn within [bl, br]
            if xn <= bl or xn >= br:
                xn = 0.5 * (bl + br)

            fxn = call_f(xn)
            if fxn is None:
                break
            if is_good(xn, fxn):
                return xn, fxn

            # Update bracket
            if fbl * fxn < 0:
                br, fbr = xn, fxn
            else:
                bl, fbl = xn, fxn

            # Also update the third point if it's better
            if abs(fxn) < abs(f3pt):
                x3pt, f3pt = xn, fxn

            if abs(br - bl) <= 2 * 2.22e-16 * max(1.0, abs(br)):
                break

        # Return the best of the three
        candidates = [(x1, f1), (x2, f2), (x3, f3)]
        candidates = [(xv, fxv) for xv, fxv in candidates if fxv is not None]
        if not candidates:
            return x1, f1
        candidates.sort(key=lambda p: abs(p[1]))
        return candidates[0][0], candidates[0][1]

    def secant(x0, x1):
        x = float(x0)
        xp = float(x1)
        fx = call_f(x)
        fxp = call_f(xp)
        if fx is None and fxp is None:
            return x, None
        if fx is None:
            x, fx = xp, fxp
        if fxp is None:
            xp, fxp = x, fx
        if is_good(x, fx):
            return x, fx
        if is_good(xp, fxp):
            return xp, fxp

        for _ in range(200):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            if fx == fxp:
                break
            xn = x - fx * (x - xp) / (fx - fxp)
            if not math.isfinite(xn):
                break
            fxn = call_f(xn)
            if fxn is None:
                break
            if is_good(xn, fxn):
                return xn, fxn
            xp, fxp = x, fx
            x, fx = xn, fxn
        if fx is not None and (fxp is None or abs(fx) < abs(fxp)):
            return x, fx
        if fxp is not None:
            return xp, fxp
        return x, fx

    def newton(x0):
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return x, None
        if is_good(x, fx):
            return x, fx

        deriv = safe_get_deriv()

        def df(xv, fxx):
            if deriv is not None:
                try:
                    d = float(deriv(xv))
                    if math.isfinite(d) and abs(d) > 1e-300:
                        return d
                except Exception:
                    pass
            h = 1e-6 * max(1.0, abs(xv))
            fx2 = call_f(xv + h)
            if fx2 is None:
                return None
            d = (fx2 - fxx) / h
            if not math.isfinite(d) or abs(d) < 1e-15:
                return None
            return d

        for _ in range(40):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            d = df(x, fx)
            if d is None:
                break
            step = fx / d
            if not math.isfinite(step):
                break
            xn = x - step
            if abs(xn - x) > 10.0 * max(1.0, abs(x)):
                xn = x + math.copysign(10.0 * max(1.0, abs(x)), xn - x)
            if not math.isfinite(xn):
                break
            fxn = call_f(xn)
            if fxn is None:
                break
            if is_good(xn, fxn):
                return xn, fxn
            if abs(fxn) < abs(fx):
                x = xn
                fx = fxn
            else:
                xh = x - 0.5 * step
                fxh = call_f(xh)
                if fxh is not None and abs(fxh) < abs(fx):
                    x = xh
                    fx = fxh
                else:
                    break
        return x, fx

    def illinois(x0, x1):
        a = float(x0)
        b = float(x1)
        fa = call_f(a)
        fb = call_f(b)
        if fa is None or fb is None:
            if fa is not None:
                return a, fa
            if fb is not None:
                return b, fb
            return a, None
        if is_good(a, fa):
            return a, fa
        if is_good(b, fb):
            return b, fb

        for _ in range(200):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            if fa * fb > 0:
                break
            if is_good(b, fb):
                return b, fb
            if abs(fa) < abs(fb):
                x = a + (b - a) * fa / (fa - fb)
                fx = call_f(x)
                if fx is None:
                    break
                if is_good(x, fx):
                    return x, fx
                b, fb = x, fx
            else:
                x = a + (b - a) * fa / (fa - fb)
                fx = call_f(x)
                if fx is None:
                    break
                if is_good(x, fx):
                    return x, fx
                a, fa = x, fx
            if abs(b - a) <= 2 * 2.22e-16 * max(1.0, abs(b)):
                break
        if fa is not None and (fb is None or abs(fa) < abs(fb)):
            return a, fa
        if fb is not None:
            return b, fb
        return a, fa

    def muller(x0, x1, x2):
        x = float(x0)
        x1 = float(x1)
        x2 = float(x2)
        fx = call_f(x)
        fx1 = call_f(x1)
        fx2 = call_f(x2)
        if fx is None and fx1 is None and fx2 is None:
            return x, None
        if fx is None:
            x, fx = x1, fx1
            x1, fx1 = x2, fx2
        if fx1 is None:
            x1, fx1 = x2, fx2
            x2, fx2 = x, fx
        if fx2 is None:
            return x, fx
        if is_good(x, fx):
            return x, fx
        if is_good(x1, fx1):
            return x1, fx1
        if is_good(x2, fx2):
            return x2, fx2

        for _ in range(100):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            d1 = x1 - x
            d2 = x2 - x
            if abs(d1) < 1e-15 or abs(d2) < 1e-15:
                break
            a1 = (fx1 - fx) / d1
            a2 = (fx2 - fx) / d2
            a3 = (a2 - a1) / (d2 - d1)
            if not math.isfinite(a3) or abs(a3) < 1e-300:
                break
            disc = a1 * a1 - 4 * a3 * fx
            if disc < 0:
                if fx != fx1:
                    h = -fx * (x - x1) / (fx - fx1)
                else:
                    break
            else:
                sqrt_disc = math.sqrt(disc)
                h1 = (-a1 + sqrt_disc) / (2 * a3)
                h2 = (-a1 - sqrt_disc) / (2 * a3)
                h = h1 if abs(h1) < abs(h2) else h2
            if not math.isfinite(h):
                break
            if abs(h) > 10.0 * max(1.0, abs(x)):
                h = math.copysign(10.0 * max(1.0, abs(x)), h)
            xn = x + h
            if not math.isfinite(xn):
                break
            fxn = call_f(xn)
            if fxn is None:
                break
            if is_good(xn, fxn):
                return xn, fxn
            x, fx = x1, fx1
            x1, fx1 = x2, fx2
            x2, fx2 = xn, fxn
        if fx is not None and (fx1 is None or abs(fx) < abs(fx1)):
            return x, fx
        if fx1 is not None:
            return x1, fx1
        return x, fx

    x0 = float(x0)

    # Phase 1: Newton from x0
    x, fx = newton(x0)
    if is_good(x, fx):
        return x
    if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
        x, fx = newton(x)
        if is_good(x, fx):
            return x

    # Phase 2: Illinois with expanding brackets
    for h in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3):
        if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 2:
            break
        x, fx = illinois(x0 - h, x0 + h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # Phase 3: Secant from nearby points
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 2:
        h = 1e-4 * max(1.0, abs(x0))
        x, fx = secant(x0, x0 + h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # Phase 4: Muller
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
        h = 1e-4 * max(1.0, abs(x0))
        x, fx = muller(x0, x0 + h, x0 - h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # Phase 5: Secant from far points
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 2:
        h = 1.0
        x, fx = secant(x0 - h, x0 + h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # Phase 6: SIQI from multiple starting configurations
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
        h = 1e-4 * max(1.0, abs(x0))
        x, fx = siqi(x0 - h, x0, x0 + h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # Phase 7: Multi-start random restarts with SIQI
    # Generate random starting points at various scales
    base_scale = max(1.0, abs(x0))
    scales = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    for scale in scales:
        if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 10:
            break
        # Try 2-3 random offsets at this scale
        for trial in range(3):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 5:
                break
            offset = rng.uniform(-scale, scale) * base_scale
            start = x0 + offset

            # Quick secant probe
            h_probe = 1e-4 * max(1.0, abs(start))
            x, fx = secant(start, start + h_probe)
            if is_good(x, fx):
                return x
            if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
                x2, fx2 = newton(x)
                if is_good(x2, fx2):
                    return x2

            # If we found something decent, try SIQI
            if fx is not None and abs(fx) < 1e-3 and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
                h_siqi = 1e-3 * max(1.0, abs(x))
                x3, fx3 = siqi(x - h_siqi, x, x + h_siqi)
                if is_good(x3, fx3):
                    return x3
                if is_close(x3, fx3) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
                    x4, fx4 = newton(x3)
                    if is_good(x4, fx4):
                        return x4

    # Final polish passes
    if is_close(best_x, best_f) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
        x, fx = newton(best_x)
        if is_good(x, fx):
            return x

    if best_f is not None and abs(best_f) < 1e-6 and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
        h = 1e-6 * max(1.0, abs(best_x))
        x, fx = secant(best_x, best_x + h)
        if is_good(x, fx):
            return x

    if best_f is not None and abs(best_f) < 1e-5 and time.monotonic() < DEADLINE and evals < MAX_EVALS - 15:
        h = 1e-4 * max(1.0, abs(best_x))
        x, fx = siqi(best_x - h, best_x, best_x + h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    return best_x
