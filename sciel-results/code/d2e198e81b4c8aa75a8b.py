"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0   # safety margin before the 120s per-eval / 150s kill
_SAFETY_MARGIN = 5.0
MAX_CALLS = 490       # stay well under 500 per equation
TOL = 1e-8


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """Return x with |f(x)| <= 1e-8 for a smooth polynomial.

    Combines Newton-with-finite-difference (fast quadratic convergence for
    smooth polynomials), the secant method, inverse-quadratic interpolation and
    bisection, all under a global wall-clock guard and a per-equation call budget.
    """
    call_count = [0]
    cache = {}

    def feval(x):
        if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
            return None
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        key = round(x, 14)
        if key in cache:
            return cache[key]
        try:
            val = f(x)
        except Exception:
            return None
        if not _is_finite(val):
            return None
        call_count[0] += 1
        cache[key] = val
        return val

    x = float(x0)
    if not math.isfinite(x):
        x = 0.0

    best_x = x
    best_abs = float('inf')
    points = []

    def record(xv, fv):
        nonlocal best_x, best_abs
        if fv is None or not math.isfinite(fv):
            return
        points.append((xv, fv))
        av = abs(fv)
        if av < best_abs:
            best_x = xv
            best_abs = av

    def done(xv, fv):
        return fv is not None and abs(fv) <= TOL

    # ---- evaluate start point, with a small local search if it is bad ----
    fx = feval(x)
    if fx is None:
        # x0 unusable; probe nearby
        for dx in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            ft = feval(x + dx)
            if ft is not None:
                x = x + dx
                fx = ft
                break
    if fx is None:
        return x
    record(x, fx)
    if done(x, fx):
        return x

    # ---- establish a secant pair near x ----
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = feval(x2)
    if fx2 is None:
        x2 = x - h
        fx2 = feval(x2)
    if fx2 is None:
        x2 = x + 1e-2 * max(1.0, abs(x))
        fx2 = feval(x2)
    if fx2 is None:
        return best_x
    record(x2, fx2)
    if done(x2, fx2):
        return x2

    # ---- Phase A: Newton with finite-difference derivative (fast) ----
    xn = best_x
    fxn = best_abs and feval(xn)
    if fxn is None:
        fxn = feval(xn)
    for _ in range(25):
        if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
            break
        if fxn is None:
            break
        if done(xn, fxn):
            return xn
        step_h = 1e-6 * max(1.0, abs(xn))
        fxp = feval(xn + step_h)
        if fxp is None:
            break
        d = (fxp - fxn) / step_h
        if not math.isfinite(d) or abs(d) < 1e-15:
            break
        xn_new = xn - fxn / d
        if not math.isfinite(xn_new):
            break
        fxn = feval(xn_new)
        if fxn is None:
            break
        record(xn_new, fxn)
        xn = xn_new
        if done(xn, fxn):
            return xn
    if best_abs <= TOL:
        return best_x

    # ---- Phase B: secant / IQUAD / bisection refinement ----
    def unique_points():
        out, seen = [], set()
        for px, pv in points:
            k = round(px, 12)
            if k not in seen:
                seen.add(k)
                out.append((px, pv))
        return out

    def find_bracket(pts):
        best = None
        best_len = float('inf')
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                xi, fi = pts[i]
                xj, fj = pts[j]
                if fi * fj < 0:
                    L = abs(xi - xj)
                    if L < best_len:
                        best_len = L
                        best = (xi, xj, fi, fj)
        return best

    bracket = find_bracket(points)
    last_x, last_fx = best_x, feval(best_x)
    if last_fx is None:
        last_fx = feval(best_x)

    for _ in range(200):
        if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
            break

        up = unique_points()
        candidate = None
        lo = hi = None
        if bracket is not None:
            lo = min(bracket[0], bracket[1])
            hi = max(bracket[0], bracket[1])

        # Inverse quadratic interpolation (3 most recent distinct points)
        if len(up) >= 3:
            x1, f1 = up[-3]
            x2, f2 = up[-2]
            x3, f3 = up[-1]
            try:
                d1 = (f1 - f2) * (f1 - f3)
                d2 = (f2 - f1) * (f2 - f3)
                d3 = (f3 - f1) * (f3 - f2)
                if abs(d1) > 1e-30 and abs(d2) > 1e-30 and abs(d3) > 1e-30:
                    xi = (x1 * (f2 * f3) / d1 +
                          x2 * (f1 * f3) / d2 +
                          x3 * (f1 * f2) / d3)
                    if math.isfinite(xi):
                        if lo is not None:
                            if lo - 1e-10 <= xi <= hi + 1e-10:
                                candidate = xi
                        elif abs(xi - last_x) < 10 * max(1.0, abs(last_x)):
                            candidate = xi
            except (ZeroDivisionError, OverflowError):
                candidate = None

        # Secant (2 most recent distinct points)
        if candidate is None and len(up) >= 2:
            x1, f1 = up[-2]
            x2, f2 = up[-1]
            if abs(f2 - f1) > 1e-30:
                xs = x2 - f2 * (x2 - x1) / (f2 - f1)
                if math.isfinite(xs):
                    if lo is not None:
                        if lo - 1e-10 <= xs <= hi + 1e-10:
                            candidate = xs
                    elif abs(xs - last_x) < 100 * max(1.0, abs(last_x)):
                        candidate = xs

        # Bisection (guaranteed if we have a bracket)
        if candidate is None and bracket is not None:
            a, b = bracket[0], bracket[1]
            mid = (a + b) / 2.0
            if abs(mid - a) < 1e-15 * max(1.0, abs(a)):
                break
            candidate = mid

        # Exploration: no interpolation and no bracket -> step outward
        if candidate is None:
            stepped = False
            for direction in (1.0, -1.0):
                if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
                    break
                step_size = 0.1 * max(1.0, abs(last_x))
                xn2 = last_x + direction * step_size
                fxn2 = feval(xn2)
                if fxn2 is not None:
                    record(xn2, fxn2)
                    last_x, last_fx = xn2, fxn2
                    if done(xn2, fxn2):
                        return xn2
                    bracket = find_bracket(points)
                    stepped = True
                    break
            if not stepped:
                break

        fx_new = feval(candidate)
        if fx_new is None:
            continue
        record(candidate, fx_new)
        if done(candidate, fx_new):
            return candidate
        last_x, last_fx = candidate, fx_new

        # Refresh the bracket from all points seen so far
        bracket = find_bracket(points)

    return best_x
# EVOLVE-BLOCK-END
