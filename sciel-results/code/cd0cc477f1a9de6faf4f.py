"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0   # safety margin before the 120s per-eval / 150s kill
_SAFETY_MARGIN = 5.0
MAX_CALLS = 480       # stay well under 500 per equation
TOL = 1e-8
TOL_POLISH = 1e-12    # tighter tolerance for final polish


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """Return x with |f(x)| <= 1e-8 for a smooth polynomial.

    Combines Newton-with-finite-difference (fast quadratic convergence),
    cubic interpolation (quartic convergence for polynomials), inverse-quadratic
    interpolation, secant, and bisection, under a global wall-clock guard and
    per-equation call budget.
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
    points = []  # list of (x, f(x)) tuples

    def record(xv, fv):
        nonlocal best_x, best_abs
        if fv is None or not math.isfinite(fv):
            return
        # Avoid duplicate x values
        for px, pv in points:
            if abs(px - xv) < 1e-14 * max(1.0, abs(xv)):
                return
        points.append((xv, fv))
        av = abs(fv)
        if av < best_abs:
            best_x = xv
            best_abs = av

    def done(fv, tol=TOL):
        return fv is not None and abs(fv) <= tol

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

    # ---- Phase A: Newton with finite-difference derivative ----
    # Fast quadratic convergence for smooth polynomials
    xn = best_x
    fxn = fx
    for _ in range(15):
        if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
            break
        if done(fxn):
            return xn
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(xn))
        fxp = feval(xn + h)
        if fxp is None:
            break
        d = (fxp - fxn) / h
        if not math.isfinite(d) or abs(d) < 1e-15:
            break
        xn_new = xn - fxn / d
        if not math.isfinite(xn_new):
            break
        # Limit step size to avoid overshooting
        if abs(xn_new - xn) > 10.0 * max(1.0, abs(xn)):
            xn_new = xn + math.copysign(10.0 * max(1.0, abs(xn)), xn_new - xn)
        fxn_new = feval(xn_new)
        if fxn_new is None:
            break
        record(xn_new, fxn_new)
        if done(fxn_new):
            return xn_new
        xn = xn_new
        fxn = fxn_new

    if best_abs <= TOL:
        return best_x

    # ---- Phase B: Cubic interpolation with fallbacks ----
    # Ensure we have enough points for cubic interpolation
    # If we don't have 4 distinct points, add more via secant steps
    while len(points) < 4 and call_count[0] < MAX_CALLS and _time_left() > 0.0:
        if len(points) >= 2:
            x1, f1 = points[-2]
            x2, f2 = points[-1]
            if abs(f2 - f1) > 1e-30:
                xs = x2 - f2 * (x2 - x1) / (f2 - f1)
                if math.isfinite(xs):
                    fxs = feval(xs)
                    if fxs is not None:
                        record(xs, fxs)
                        if done(fxs):
                            return xs
                        continue
        # If secant fails, step outward
        h = 1e-3 * max(1.0, abs(best_x))
        xp = best_x + h
        fxp = feval(xp)
        if fxp is not None:
            record(xp, fxp)
            if done(fxp):
                return xp
        else:
            xp = best_x - h
            fxp = feval(xp)
            if fxp is not None:
                record(xp, fxp)
                if done(fxp):
                    return xp
            else:
                break

    def cubic_interp(x1, f1, x2, f2, x3, f3, x4, f4):
        """Cubic Lagrange interpolation to find root estimate."""
        try:
            xs = [x1, x2, x3, x4]
            fs = [f1, f2, f3, f4]
            
            # Check for nearly identical x values
            for i in range(4):
                for j in range(i+1, 4):
                    if abs(xs[i] - xs[j]) < 1e-13 * max(1.0, abs(xs[i])):
                        return None
            
            # Build cubic polynomial coefficients
            a3 = 0.0
            a2 = 0.0
            a1 = 0.0
            a0 = 0.0
            
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
                a1 += coeff * (oa*ob + oa*oc + ob*oc)
                a0 += coeff * (-(oa*ob*oc))
            
            # Find root of cubic P(x) = a3*x^3 + a2*x^2 + a1*x + a0
            if abs(a3) < 1e-30:
                if abs(a2) < 1e-30:
                    if abs(a1) < 1e-30:
                        return None
                    return -a0 / a1
                disc = a1*a1 - 4*a2*a0
                if disc < 0:
                    return None
                sqrt_disc = math.sqrt(disc)
                r1 = (-a1 + sqrt_disc) / (2*a2)
                r2 = (-a1 - sqrt_disc) / (2*a2)
                if abs(r1 - best_x) < abs(r2 - best_x):
                    return r1
                else:
                    return r2
            
            # Newton's method on the cubic
            start_x = best_x
            for _ in range(10):
                p_val = a3*start_x**3 + a2*start_x**2 + a1*start_x + a0
                p_deriv = 3*a3*start_x**2 + 2*a2*start_x + a1
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
    for _ in range(40):
        if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
            break
        if best_abs <= TOL:
            break

        candidate = None
        lo = hi = None
        if bracket is not None:
            lo = min(bracket[0], bracket[1])
            hi = max(bracket[0], bracket[1])

        # Try cubic interpolation with 4 most recent distinct points
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

        # Fallback: IQUAD with 3 points
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

        # Fallback: secant with 2 points
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

        # Fallback: bisection if we have a bracket
        if candidate is None and bracket is not None:
            a, b = bracket
            mid = (a + b) / 2.0
            if abs(mid - a) < 1e-15 * max(1.0, abs(a)):
                break
            candidate = mid

        # Exploration: step outward
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

    # ---- Phase C: Final high-precision polish ----
    # If we're close but not quite there, do a few more Newton steps with tighter tol
    if best_abs <= 1e-4 and call_count[0] < MAX_CALLS and _time_left() > 0.0:
        xn = best_x
        fxn = feval(xn)
        if fxn is not None:
            for _ in range(10):
                if call_count[0] >= MAX_CALLS or _time_left() <= 0.0:
                    break
                if abs(fxn) <= TOL_POLISH:
                    return xn
                h = 1e-7 * max(1.0, abs(xn))
                fxp = feval(xn + h)
                if fxp is None:
                    break
                d = (fxp - fxn) / h
                if not math.isfinite(d) or abs(d) < 1e-18:
                    break
                xn_new = xn - fxn / d
                if not math.isfinite(xn_new):
                    break
                fxn_new = feval(xn_new)
                if fxn_new is None:
                    break
                record(xn_new, fxn_new)
                if abs(fxn_new) < abs(fxn):
                    xn = xn_new
                    fxn = fxn_new
                else:
                    break

    return best_x
# EVOLVE-BLOCK-END
