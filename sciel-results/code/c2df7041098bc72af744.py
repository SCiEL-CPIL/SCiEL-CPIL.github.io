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

    Uses cubic interpolation (quartic convergence for polynomials) as primary
    method, with secant and bisection as fallbacks, under a global wall-clock
    guard and per-equation call budget.
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
    points = []  # list of (x, f(x)) tuples, most recent last

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

    # ---- establish initial points for interpolation ----
    # We need at least 3-4 distinct points for cubic interpolation
    h_base = 1e-3 * max(1.0, abs(x))
    for h in [h_base, -h_base, 2*h_base, -2*h_base, 0.5*h_base, -0.5*h_base,
              5*h_base, -5*h_base, 10*h_base, -10*h_base]:
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

    if len(points) < 2:
        return best_x

    def cubic_interp(x1, f1, x2, f2, x3, f3, x4, f4):
        """Cubic Lagrange interpolation to find root estimate."""
        try:
            # Check for nearly identical x values
            for i in range(4):
                for j in range(i+1, 4):
                    xs = [x1, x2, x3, x4]
                    if abs(xs[i] - xs[j]) < 1e-13 * max(1.0, abs(xs[i])):
                        return None
            # Lagrange cubic interpolation
            # P(x) = sum_i f(xi) * L_i(x)
            # We want to find x where P(x) = 0
            # This gives a cubic equation in x
            # Use Newton's method on the cubic, or solve directly
            # Alternative: use the fact that for a cubic through 4 points,
            # we can set up the interpolation and find its root
            
            # Simpler approach: use the 4 points to estimate the next point
            # via inverse cubic interpolation or direct cubic root finding
            # Here we use a weighted approach similar to IQUAD but with 4 points
            
            # Actually, let's use the standard cubic interpolation formula
            # and find where it crosses zero using a few Newton steps
            # on the cubic polynomial itself.
            
            # Build the cubic polynomial coefficients using Lagrange basis
            # P(t) where t is the parameter, but we work in x directly
            # P(x) = sum_{i=0}^{3} f_i * prod_{j!=i} (x - x_j)/(x_i - x_j)
            # This is a cubic in x. Let's find its root.
            
            xs = [x1, x2, x3, x4]
            fs = [f1, f2, f3, f4]
            
            # Coefficients of the cubic: P(x) = a3*x^3 + a2*x^2 + a1*x + a0
            # Using Lagrange interpolation
            a3 = 0.0
            a2 = 0.0
            a1 = 0.0
            a0 = 0.0
            
            for i in range(4):
                # L_i(x) = prod_{j!=i} (x - x_j)/(x_i - x_j)
                # This is a quadratic in x: L_i(x) = (x - x_j1)(x - x_j2)/(x_i - x_j1)(x_i - x_j2)
                # where j1, j2 are the other two indices
                others = [j for j in range(4) if j != i]
                j1, j2 = others[0], others[1]
                denom = (xs[i] - xs[j1]) * (xs[i] - xs[j2])
                if abs(denom) < 1e-30:
                    return None
                # L_i(x) = (x - x_j1)(x - x_j2) / denom
                # = (x^2 - x*(x_j1+x_j2) + x_j1*x_j2) / denom
                coeff = fs[i] / denom
                a3 += coeff
                a2 += coeff * (-(xs[j1] + xs[j2]))
                a1 += coeff * (xs[j1] * xs[j2])
            
            # Wait, that's wrong. L_i is quadratic, so P is cubic.
            # Let me redo: each L_i(x) is quadratic, so P(x) = sum f_i * L_i(x) is cubic.
            # L_i(x) = prod_{j!=i} (x - x_j)/(x_i - x_j)
            # For 4 points, each L_i is a product of 2 linear terms = quadratic.
            
            # Let me compute coefficients properly
            a3 = 0.0
            a2 = 0.0
            a1 = 0.0
            a0 = 0.0
            
            for i in range(4):
                others = [j for j in range(4) if j != i]
                j1, j2 = others[0], others[1]
                denom = (xs[i] - xs[j1]) * (xs[i] - xs[j2])
                if abs(denom) < 1e-30:
                    return None
                coeff = fs[i] / denom
                # L_i(x) = (x - x_j1)*(x - x_j2) * coeff
                # = (x^2 - (x_j1+x_j2)*x + x_j1*x_j2) * coeff
                a3 += coeff
                a2 += coeff * (-(xs[j1] + xs[j2]))
                a1 += coeff * (xs[j1] * xs[j2])
            
            # P(x) = a3*x^3 + a2*x^2 + a1*x + a0  (a0 = 0 since no constant term added)
            # Wait, I'm missing the constant. Let me re-examine.
            # L_i(x) = (x-x_j1)(x-x_j2) / denom
            # = [x^2 - (x_j1+x_j2)x + x_j1*x_j2] / denom
            # So P(x) = sum_i f_i/denom_i * [x^2 - (x_j1+x_j2)x + x_j1*x_j2]
            # This is only quadratic! That can't be right for 4 points.
            
            # Oh wait, for 4 points, each L_i is a product of 3 linear terms, not 2!
            # L_i(x) = prod_{j!=i, j=0..3} (x - x_j)/(x_i - x_j)
            # With 4 points, there are 3 other points, so L_i is cubic.
            
            a3 = 0.0
            a2 = 0.0
            a1 = 0.0
            a0 = 0.0
            
            for i in range(4):
                others = [j for j in range(4) if j != i]
                # 3 other points
                denom = 1.0
                for j in others:
                    denom *= (xs[i] - xs[j])
                if abs(denom) < 1e-30:
                    return None
                coeff = fs[i] / denom
                # L_i(x) = prod_{j in others} (x - x_j) * coeff
                # Expand prod (x - x_j) for 3 terms
                # = (x - a)(x - b)(x - c) = x^3 - (a+b+c)x^2 + (ab+ac+bc)x - abc
                oa, ob, oc = xs[others[0]], xs[others[1]], xs[others[2]]
                a3_i = coeff
                a2_i = coeff * (-(oa + ob + oc))
                a1_i = coeff * (oa*ob + oa*oc + ob*oc)
                a0_i = coeff * (-(oa*ob*oc))
                a3 += a3_i
                a2 += a2_i
                a1 += a1_i
                a0 += a0_i
            
            # Now P(x) = a3*x^3 + a2*x^2 + a1*x + a0
            # Find root of this cubic. Use Newton's method starting from best_x
            if abs(a3) < 1e-30:
                # Degenerate to quadratic or lower
                if abs(a2) < 1e-30:
                    if abs(a1) < 1e-30:
                        return None
                    return -a0 / a1
                # Quadratic: a2*x^2 + a1*x + a0 = 0
                disc = a1*a1 - 4*a2*a0
                if disc < 0:
                    return None
                sqrt_disc = math.sqrt(disc)
                r1 = (-a1 + sqrt_disc) / (2*a2)
                r2 = (-a1 - sqrt_disc) / (2*a2)
                # Pick the root closer to best_x
                if abs(r1 - best_x) < abs(r2 - best_x):
                    return r1
                else:
                    return r2
            
            # Cubic: use Newton's method
            # Start from the point with smallest |f|
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

    return best_x
# EVOLVE-BLOCK-END
