"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0   # safety margin before the 120s per-eval / 150s kill
_SAFETY_MARGIN = 5.0
MAX_CALLS = 495       # stay well under 500 per equation
TOL = 1e-8


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _polynomial_roots(f, x0, feval, record, done, points):
    """Direct coefficient recovery: fit a low-degree polynomial to a set of
    well-spread points, solve for the real root nearest x0, and polish it.
    Returns (root, fval) on success or (None, None)."""
    try:
        import numpy as np
    except Exception:
        return None, None

    if len(points) < 4:
        return None, None

    # Choose well-spread, distinct x values to build a stable Vandermonde.
    cand = list(points)
    if len(cand) < 4:
        return None, None
    cand.sort(key=lambda p: p[0])
    # Prefer the extreme points plus a couple interior ones for conditioning.
    chosen = [cand[0], cand[-1]]
    m = len(cand)
    for k in (m // 3, (2 * m) // 3):
        if 0 <= k < m and cand[k] not in chosen:
            chosen.append(cand[k])
    # Top up if we don't have enough distinct points.
    if len(chosen) < 4:
        for p in cand:
            if p not in chosen:
                chosen.append(p)
            if len(chosen) >= 4:
                break
    if len(chosen) < 4:
        return None, None
    chosen = chosen[:4]

    xs = [p[0] for p in chosen]
    ys = [p[1] for p in chosen]

    # Solve for coefficients of a cubic: f(x) ~= c0 + c1 x + c2 x^2 + c3 x^3
    try:
        V = np.vander(xs, N=4, increasing=True)
        coeffs = np.linalg.solve(V, ys)
    except Exception:
        return None, None

    if not np.all(np.isfinite(coeffs)):
        return None, None

    # Find real roots of the fitted cubic.
    try:
        rts = np.roots(coeffs[::-1])  # np.roots expects descending powers
    except Exception:
        return None, None

    real_roots = [r.real for r in rts if abs(r.imag) <= 1e-6 * max(1.0, abs(r.real))]
    if not real_roots:
        return None, None

    # Pick the real root closest to the current best x.
    best = min(real_roots, key=lambda r: abs(r - x0))

    # Polish this candidate with a couple of secant steps to reach 1e-8.
    bx = float(best)
    fb = feval(bx)
    if fb is None:
        return None, None
    record(bx, fb)
    if done(fb):
        return bx, fb

    # Secant polish using the two nearest known points.
    near = sorted(points, key=lambda p: abs(p[0] - bx))
    if len(near) >= 2:
        x1, f1 = near[0]
        x2, f2 = near[1]
        for _ in range(12):
            if call_budget_exhausted():
                break
            if abs(f2 - f1) < 1e-30:
                break
            xc = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not _is_finite(xc):
                break
            fc = feval(xc)
            if fc is None:
                break
            record(xc, fc)
            if done(fc):
                return xc, fc
            x1, f1 = x2, f2
            x2, f2 = xc, fc

    # Return best polished value.
    return bx, fb


def call_budget_exhausted():
    return _call_count[0] >= MAX_CALLS or _time_left() <= 0.0


_call_count = [0]


def find_root(f, x0):
    """Return x with |f(x)| <= 1e-8 for a smooth polynomial.

    Combines direct polynomial-coefficient recovery (numpy.roots) with
    cubic/inverse-quadratic/secant interpolation and bisection fallbacks,
    under a global wall-clock guard and per-equation call budget.
    """
    _call_count[0] = 0
    call_count = _call_count
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
    points = []  # list of (x, f(x)) tuples, most recent last

    def record(xv, fv):
        nonlocal best_x, best_abs
        if fv is None or not math.isfinite(fv):
            return
        for px, pv in points:
            if abs(px - xv) < 1e-14 * max(1.0, abs(xv)):
                return
        points.append((xv, fv))
        av = abs(fv)
        if av < best_abs:
            best_x = xv
            best_abs = av

    def done(fv):
        return fv is not None and abs(fv) <= TOL

    # ---- evaluate start point ----
    fx = feval(x)
    if fx is None:
        for dx in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            ft = feval(x + dx)
            if ft is not None:
                x = x + dx
                fx = ft
                break
    if fx is None:
        return x
    record(x, fx)
    if done(fx):
        return x

    # ---- establish initial points for interpolation / polynomial fit ----
    # We need at least 4 distinct, reasonably-spread points for a cubic fit.
    # Use geometrically increasing steps so the Vandermonde is well-conditioned.
    scale = max(1.0, abs(x))
    for h in [1e-2 * scale, -1e-2 * scale, 1e-1 * scale, -1e-1 * scale,
              1.0 * scale, -1.0 * scale, 10.0 * scale, -10.0 * scale,
              0.1 * scale, -0.1 * scale, 100.0 * scale, -100.0 * scale]:
        if len(points) >= 4:
            break
        if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
            break
        xp = x + h
        fxp = feval(xp)
        if fxp is not None:
            record(xp, fxp)
            if done(fxp):
                return xp

    # ---- DIRECT POLYNOMIAL COEFFICIENT RECOVERY (primary, low call cost) ----
    if len(points) >= 4 and _time_left() > 1.0:
        root, fval = _polynomial_roots(f, x0, feval, record, done, points)
        if root is not None and fval is not None and abs(fval) <= TOL:
            return root
        # Even if not yet at tolerance, keep the best found so far.
        if root is not None and fval is not None and abs(fval) < best_abs:
            best_x = root
            best_abs = abs(fval)
            # Polish the recovered root with Newton's method using finite-difference derivative
            if call_count[0] < MAX_CALLS - 5 and _time_left() > 0.5:
                for _ in range(8):
                    h = 1e-8 * max(1.0, abs(best_x))
                    fp = feval(best_x + h)
                    fm = feval(best_x - h)
                    if fp is None or fm is None:
                        break
                    deriv = (fp - fm) / (2 * h)
                    if abs(deriv) < 1e-30:
                        break
                    step = fval / deriv
                    new_x = best_x - step
                    if not math.isfinite(new_x):
                        break
                    new_f = feval(new_x)
                    if new_f is None:
                        break
                    if abs(new_f) < best_abs:
                        best_x = new_x
                        best_abs = abs(new_f)
                        fval = new_f
                        if best_abs <= TOL:
                            return best_x
                    else:
                        break

    if len(points) < 2:
        return best_x

    def cubic_interp(x1, f1, x2, f2, x3, f3, x4, f4):
        """Cubic Lagrange interpolation to find root estimate."""
        try:
            xs = [x1, x2, x3, x4]
            fs = [f1, f2, f3, f4]
            for i in range(4):
                for j in range(i + 1, 4):
                    if abs(xs[i] - xs[j]) < 1e-13 * max(1.0, abs(xs[i])):
                        return None
            a3 = a2 = a1 = a0 = 0.0
            for i in range(4):
                others = [j for j in range(4) if j != i]
                denom = 1.0
                for j in others:
                    denom *= (xs[i] - xs[j])
                if abs(denom) < 1e-30:
                    return None
                coeff = fs[i] / denom
                oa, ob, oc = xs[others[0]], xs[others[1]], xs[others[2]]
                a3 += coeff
                a2 += coeff * (-(oa + ob + oc))
                a1 += coeff * (oa * ob + oa * oc + ob * oc)
                a0 += coeff * (-(oa * ob * oc))
            if abs(a3) < 1e-30:
                if abs(a2) < 1e-30:
                    if abs(a1) < 1e-30:
                        return None
                    return -a0 / a1
                disc = a1 * a1 - 4 * a2 * a0
                if disc < 0:
                    return None
                sqrt_disc = math.sqrt(disc)
                r1 = (-a1 + sqrt_disc) / (2 * a2)
                r2 = (-a1 - sqrt_disc) / (2 * a2)
                return r1 if abs(r1 - best_x) < abs(r2 - best_x) else r2
            start_x = best_x
            for _ in range(10):
                p_val = a3 * start_x**3 + a2 * start_x**2 + a1 * start_x + a0
                p_deriv = 3 * a3 * start_x**2 + 2 * a2 * start_x + a1
                if abs(p_deriv) < 1e-30:
                    break
                start_x_new = start_x - p_val / p_deriv
                if not math.isfinite(start_x_new):
                    break
                if abs(start_x_new - start_x) < 1e-14 * max(1.0, abs(start_x)):
                    break
                start_x = start_x_new
            return start_x if math.isfinite(start_x) else None
        except (ZeroDivisionError, OverflowError, ArithmeticError):
            return None

    def iquad_interp(x1, f1, x2, f2, x3, f3):
        """Inverse quadratic interpolation."""
        try:
            d1 = (f1 - f2) * (f1 - f3)
            d2 = (f2 - f1) * (f2 - f3)
            d3 = (f3 - f1) * (f3 - f2)
            if abs(d1) < 1e-30 or abs(d2) < 1e-30 or abs(d3) < 1e-30:
                return None
            xi = (x1 * (f2 * f3) / d1 +
                  x2 * (f1 * f3) / d2 +
                  x3 * (f1 * f2) / d3)
            return xi if math.isfinite(xi) else None
        except (ZeroDivisionError, OverflowError):
            return None

    def secant_interp(x1, f1, x2, f2):
        """Secant method step."""
        try:
            if abs(f2 - f1) < 1e-30:
                return None
            xs = x2 - f2 * (x2 - x1) / (f2 - f1)
            return xs if math.isfinite(xs) else None
        except (ZeroDivisionError, OverflowError):
            return None

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
                        best = (xi, xj)
        return best

    bracket = find_bracket(points)

    # ---- Main iteration: cubic interpolation with fallbacks ----
    for _ in range(60):
        if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
            break
        if best_abs <= TOL:
            break

        candidate = None
        lo = hi = None
        if bracket is not None:
            lo = min(bracket[0], bracket[1])
            hi = max(bracket[0], bracket[1])

        if len(points) >= 4:
            x1, f1 = points[-4]
            x2, f2 = points[-3]
            x3, f3 = points[-2]
            x4, f4 = points[-1]
            candidate = cubic_interp(x1, f1, x2, f2, x3, f3, x4, f4)
            if candidate is not None and math.isfinite(candidate):
                if lo is not None:
                    if not (lo - 1e-10 <= candidate <= hi + 1e-10):
                        candidate = None
                elif abs(candidate - best_x) > 100 * max(1.0, abs(best_x)):
                    candidate = None

        if candidate is None and len(points) >= 3:
            x1, f1 = points[-3]
            x2, f2 = points[-2]
            x3, f3 = points[-1]
            candidate = iquad_interp(x1, f1, x2, f2, x3, f3)
            if candidate is not None and math.isfinite(candidate):
                if lo is not None:
                    if not (lo - 1e-10 <= candidate <= hi + 1e-10):
                        candidate = None
                elif abs(candidate - best_x) > 100 * max(1.0, abs(best_x)):
                    candidate = None

        if candidate is None and len(points) >= 2:
            x1, f1 = points[-2]
            x2, f2 = points[-1]
            candidate = secant_interp(x1, f1, x2, f2)
            if candidate is not None and math.isfinite(candidate):
                if lo is not None:
                    if not (lo - 1e-10 <= candidate <= hi + 1e-10):
                        candidate = None
                elif abs(candidate - best_x) > 100 * max(1.0, abs(best_x)):
                    candidate = None

        if candidate is None and bracket is not None:
            a, b = bracket
            mid = (a + b) / 2.0
            if abs(mid - a) < 1e-15 * max(1.0, abs(a)):
                break
            candidate = mid

        if candidate is None:
            stepped = False
            for direction in (1.0, -1.0):
                if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
                    break
                step_size = 0.1 * max(1.0, abs(best_x))
                xn2 = best_x + direction * step_size
                fxn2 = feval(xn2)
                if fxn2 is not None:
                    record(xn2, fxn2)
                    if done(fxn2):
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
        if done(fx_new):
            return candidate

        bracket = find_bracket(points)

    # ---- Final high-precision polish pass ----
    if best_abs <= 1e-3 and call_count[0] < MAX_CALLS - 5 and _time_left() > 1.0:
        # Try Newton's method with finite-difference derivative for faster convergence
        for _ in range(15):
            if call_count[0] >= MAX_CALLS - 1 or _time_left() <= 0.5:
                break
            if best_abs <= TOL:
                break
            h = 1e-8 * max(1.0, abs(best_x))
            fp = feval(best_x + h)
            fm = feval(best_x - h)
            if fp is None or fm is None:
                break
            deriv = (fp - fm) / (2 * h)
            if abs(deriv) < 1e-30:
                break
            fcur = feval(best_x)
            if fcur is None:
                break
            step = fcur / deriv
            new_x = best_x - step
            if not math.isfinite(new_x):
                break
            new_f = feval(new_x)
            if new_f is None:
                break
            if abs(new_f) < best_abs:
                best_x = new_x
                best_abs = abs(new_f)
                if best_abs <= TOL:
                    return best_x
            else:
                # Newton step didn't help; fall back to secant-like step
                break

        # If still not at tolerance, do a few more secant steps
        if best_abs > TOL and len(points) >= 2 and call_count[0] < MAX_CALLS - 3 and _time_left() > 0.5:
            near = sorted(points, key=lambda p: abs(p[0] - best_x))
            x1, f1 = near[0]
            x2, f2 = near[1]
            for _ in range(8):
                if call_count[0] >= MAX_CALLS - 1 or _time_left() <= 0.3:
                    break
                if best_abs <= TOL:
                    break
                if abs(f2 - f1) < 1e-30:
                    break
                xc = x2 - f2 * (x2 - x1) / (f2 - f1)
                if not _is_finite(xc):
                    break
                fc = feval(xc)
                if fc is None:
                    break
                record(xc, fc)
                if done(fc):
                    return xc
                if abs(fc) < best_abs:
                    best_x = xc
                    best_abs = abs(fc)
                x1, f1 = x2, f2
                x2, f2 = xc, fc

    return best_x
# EVOLVE-BLOCK-END
