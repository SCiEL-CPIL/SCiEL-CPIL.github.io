"""Root finding: Ill-conditioned - model-and-optimize with global minimization fallback."""
import math
import time

import numpy as np

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.

    MODEL-AND-OPTIMIZE approach:
      1. Sample f around x0 (adaptive, log-spaced both directions) -> build local model.
      2. Fit polynomial coefficients (least squares) to the samples.
      3. Optimize the model: find roots of the fitted polynomial.
      4. Polish each model root with the real f (secant/Newton) to hit |f|<=1e-8.
      5. If that fails, try scipy.optimize.minimize_scalar on f(x)^2 in a bounded region.
      6. Keep the best point found; deadline-guarded.
    """
    start_time = time.time()
    MAX_WALL = 118.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        eval_count[0] += 1
        val = f(x)
        if not math.isfinite(val):
            val = 0.0
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        return val

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def check_budget():
        return (eval_count[0] < MAX_EVALS) and (time.time() - start_time < MAX_WALL)

    def polish(x0, fx0=None, max_iter=60):
        """Secant + damped-Newton polish on the real f from a near-root."""
        if not check_budget():
            return None
        x = float(x0)
        if fx0 is None:
            fx0 = eval_f(x)
        fx = fx0
        if is_good(x, fx):
            return x
        best_xl = x
        best_fl = abs(fx)

        # build a second close point
        h = 1e-3 * max(1.0, abs(x))
        x_prev = x - h
        fx_prev = eval_f(x_prev)
        if is_good(x_prev, fx_prev):
            return x_prev

        for _ in range(max_iter):
            if not check_budget():
                break
            if fx == fx_prev:
                break
            d = (fx - fx_prev) / (x - x_prev)
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            # damp large steps
            if abs(x_new - x) > 1e2 * max(1.0, abs(x)):
                x_new = x + 1e2 * max(1.0, abs(x)) * (1 if (x_new - x) > 0 else -1)
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            if abs(fx_new) < best_fl:
                best_fl = abs(fx_new)
                best_xl = x_new
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
            if abs(x_new - x_prev) < 1e-14 * max(1.0, abs(x_new)):
                break
        return best_xl if best_fl <= TARGET_TOL else None

    def brentq_counted(a, b, fa, fb):
        """brentq on real f with counted evals; fallback bisection."""
        if not check_budget():
            return None
        if fa * fb > 0:
            return None
        try:
            from scipy.optimize import brentq
            def fw(x):
                return eval_f(x)
            r = brentq(fw, a, b, xtol=1e-12, rtol=1e-12, maxiter=60)
            fr = eval_f(r)
            if is_good(r, fr):
                return r
            if abs(fr) <= 1e-6:
                return polish(r, fr)
        except Exception:
            pass
        # bisection fallback
        if not check_budget():
            return None
        if fa * fb > 0:
            return None
        left, right, fl, fr = a, b, fa, fb
        best_xl = left if abs(fl) < abs(fr) else right
        best_fl = min(abs(fl), abs(fr))
        for _ in range(80):
            if not check_budget():
                break
            mid = (left + right) / 2
            fm = eval_f(mid)
            if is_good(mid, fm):
                return mid
            if abs(fm) < best_fl:
                best_fl = abs(fm)
                best_xl = mid
            if fl * fm <= 0:
                right, fr = mid, fm
            else:
                left, fl = mid, fm
            if abs(right - left) < 1e-14 * max(1.0, abs(mid)):
                break
        return best_xl if best_fl <= TARGET_TOL else None

    # ---- MODEL-AND-OPTIMIZE: sample, fit, optimize model roots ----
    def sample_and_model(x0):
        """Return (xs, fs) samples around x0, then fit polynomial coeffs."""
        xs = []
        fs = []

        def add(x):
            if math.isfinite(x) and x not in xs:
                xs.append(x)

        base = max(1e-6, abs(x0)) if x0 != 0 else 1e-6
        scale = max(1.0, abs(x0))
        for s in [0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0,
                  10.0, -10.0, 30.0, -30.0, 100.0, -100.0]:
            add(x0 + s * scale)
        for s in [1e-2, -1e-2, 1e-3, -1e-3, 1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6]:
            add(x0 + s * scale)

        for x in xs:
            if not check_budget():
                break
            fx = eval_f(x)
            if is_good(x, fx):
                return x, xs, fs
            fs.append(fx)
        return None, xs, fs

    def model_roots(xs, fs):
        """Fit polynomial to (xs, fs) and return real roots of the fitted model."""
        if len(xs) < 2:
            return []
        xc = (max(xs) + min(xs)) / 2.0
        sc = max(1e-12, (max(xs) - min(xs)) / 2.0)
        u = [(x - xc) / sc for x in xs]
        y = list(fs)
        n = len(u)
        deg = min(n - 1, 12)
        try:
            coeffs = np.polyfit(u, y, deg)
        except Exception:
            try:
                coeffs = np.polyfit(u, y, min(n - 1, 6))
            except Exception:
                return []
        try:
            r = np.roots(coeffs)
        except Exception:
            return []
        out = []
        for ri in r:
            if abs(ri.imag) > 1e-7 * max(1.0, abs(ri.real)):
                continue
            x_real = ri.real * sc + xc
            if math.isfinite(x_real):
                out.append(float(x_real))
        return out

    # ---- MAIN ----
    # 1) immediate check
    fx0 = eval_f(x0)
    if is_good(x0, fx0):
        return x0

    # 2) sample + fit + optimize model roots
    found, xs, fs = sample_and_model(x0)
    if found is not None:
        return found

    candidates = []
    if len(xs) >= 3:
        candidates.extend(model_roots(xs, fs))
    for x in xs:
        candidates.append(x)
    candidates.append(float(x0))

    seen = set()
    uniq = []
    for c in candidates:
        key = round(c, 12)
        if key in seen or not math.isfinite(c):
            continue
        seen.add(key)
        uniq.append(c)

    for c in uniq:
        if not check_budget():
            break
        r = polish(c)
        if r is not None:
            return r

    # 3) bracket search + brentq fallback
    def find_bracket(x0):
        if not check_budget():
            return None
        fx0 = eval_f(x0)
        if is_good(x0, fx0):
            return (x0, x0, fx0, fx0)
        base_h = 1e-2 * max(1.0, abs(x0))
        for scale in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000, 1000000]:
            if not check_budget():
                break
            h = base_h * scale
            xl, xr = x0 - h, x0 + h
            fxl = eval_f(xl)
            if is_good(xl, fxl):
                return (xl, xl, fxl, fxl)
            fxr = eval_f(xr)
            if is_good(xr, fxr):
                return (xr, xr, fxr, fxr)
            if fxl * fxr < 0:
                return (xl, xr, fxl, fxr)
            if fxl * fx0 < 0:
                return (xl, x0, fxl, fx0)
            if fx0 * fxr < 0:
                return (x0, xr, fx0, fxr)
        return None

    bracket = find_bracket(x0)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a == b:
            return a
        if fa * fb < 0:
            r = brentq_counted(a, b, fa, fb)
            if r is not None:
                return r

    # 4) scan for a sign change locally
    if check_budget():
        base_h = 1e-3 * max(1.0, abs(x0))
        prev_x = x0 - base_h
        prev_fx = eval_f(prev_x)
        for i in range(1, 200):
            if not check_budget():
                break
            cur_x = x0 + (i - 1) * base_h
            cur_fx = eval_f(cur_x)
            if is_good(cur_x, cur_fx):
                return cur_x
            if prev_fx * cur_fx < 0:
                r = brentq_counted(prev_x, cur_x, prev_fx, cur_fx)
                if r is not None:
                    return r
            prev_x, prev_fx = cur_x, cur_fx

    # 5) Global minimization of f(x)^2 using scipy.optimize.minimize_scalar
    if check_budget():
        try:
            from scipy.optimize import minimize_scalar
            # Define objective as f(x)^2
            def obj(x):
                val = eval_f(x)
                return val * val

            # Choose a reasonable bounded region around x0
            scale = max(1.0, abs(x0))
            bounds = (x0 - 100 * scale, x0 + 100 * scale)
            res = minimize_scalar(obj, bounds=bounds, method='bounded',
                                   options={'xatol': 1e-10, 'maxiter': 100})
            if res.success and check_budget():
                x_min = res.x
                f_min = eval_f(x_min)
                if is_good(x_min, f_min):
                    return x_min
                if abs(f_min) <= 1e-6:
                    r = polish(x_min, f_min)
                    if r is not None:
                        return r
        except Exception:
            pass

    return best_x[0]

# EVOLVE-BLOCK-END
