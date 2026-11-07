import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Strategy:
    1. Check if x0 is already a root.
    2. Try secant/brent on a small bracket around x0 using adaptive bracketing.
    3. Use polynomial reconstruction (quadratic then cubic) as a secondary approach.
    4. Fall back to safeguarded Newton / Brent with wider brackets.
    """
    TOL = 1e-8
    TOL_POLISH = 1e-12
    DEADLINE = 115.0
    START_TIME = time.time()

    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    def polish_root(x, fx_val, max_iter=12):
        """Final high-precision polish using Newton's method with tight tolerance."""
        if not is_finite(x):
            return x, fx_val
        x_cur = x
        f_cur = fx_val
        for _ in range(max_iter):
            if not check_deadline():
                break
            if abs(f_cur) <= TOL_POLISH:
                return x_cur, f_cur
            h = 1e-8 * max(1.0, abs(x_cur))
            try:
                f_plus = f(x_cur + h)
                f_minus = f(x_cur - h)
            except Exception:
                break
            if not (is_finite(f_plus) and is_finite(f_minus)):
                break
            df = (f_plus - f_minus) / (2 * h)
            if abs(df) < 1e-15:
                break
            step = f_cur / df
            x_new = x_cur - step
            if not is_finite(x_new):
                break
            if abs(step) > 100 * max(1.0, abs(x_cur)):
                step = 0.5 * step
                x_new = x_cur - step
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not is_finite(f_new):
                break
            x_cur = x_new
            f_cur = f_new
        return x_cur, f_cur

    # Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0

    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)

    # Strategy 1: Adaptive bracketing with Brent's method
    # Try small brackets first, then expand
    for step in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6, 1e7, 1e8]:
        if not check_deadline():
            break

        a = x0 - step
        b = x0 + step

        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            continue

        if not (is_finite(fa) and is_finite(fb)):
            continue

        if abs(fa) < best_f_abs:
            best_f_abs = abs(fa)
            best_x = a
        if abs(fb) < best_f_abs:
            best_f_abs = abs(fb)
            best_x = b

        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b

        if fa * fb < 0:
            try:
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        root, f_root = polish_root(root, f_root)
                        if is_finite(root) and abs(f_root) <= TOL:
                            return root
                    return root
            except Exception:
                pass

    # Strategy 2: Quadratic fit with 3 points
    try:
        h = 1.0
        xq1 = x0 - h
        xq2 = x0
        xq3 = x0 + h

        fq1 = f(xq1)
        fq3 = f(xq3)

        if all(math.isfinite(v) for v in [fq1, fx0, fq3]):
            Vq = np.vander([xq1, xq2, xq3], N=3, increasing=True)
            try:
                coeff_q = np.linalg.solve(Vq, np.array([fq1, fx0, fq3]))
                c, b, a = coeff_q[0], coeff_q[1], coeff_q[2]

                if abs(a) > 1e-15:
                    disc = b * b - 4 * a * c
                    if disc >= 0:
                        sqrt_disc = math.sqrt(disc)
                        r1 = (-b + sqrt_disc) / (2 * a)
                        r2 = (-b - sqrt_disc) / (2 * a)
                        candidates = [r1, r2]
                    else:
                        candidates = []
                else:
                    if abs(b) > 1e-15:
                        candidates = [-c / b]
                    else:
                        candidates = []

                if candidates:
                    best_r = min(candidates, key=lambda r: abs(r - x0))
                    f_best = f(best_r)
                    if math.isfinite(f_best) and abs(f_best) <= TOL:
                        best_r, f_best = polish_root(best_r, f_best)
                        if is_finite(best_r) and abs(f_best) <= TOL:
                            return best_r
            except np.linalg.LinAlgError:
                pass
    except Exception:
        pass

    # Strategy 3: Cubic fit with 4 points
    try:
        h = 1.0
        x1 = x0 - 1.5 * h
        x2 = x0 - 0.5 * h
        x3 = x0 + 0.5 * h
        x4 = x0 + 1.5 * h

        f1 = f(x1)
        f2 = f(x2)
        f3 = f(x3)
        f4 = f(x4)

        if all(math.isfinite(v) for v in [f1, f2, f3, f4]):
            V = np.vander([x1, x2, x3, x4], N=4, increasing=True)
            try:
                coeff_vec = np.linalg.solve(V, np.array([f1, f2, f3, f4]))
                d, c, b, a = coeff_vec[0], coeff_vec[1], coeff_vec[2], coeff_vec[3]

                roots = np.roots([a, b, c, d])

                real_roots = []
                for r in roots:
                    if abs(r.imag) < 1e-10:
                        real_roots.append(r.real)

                if real_roots:
                    best_r = min(real_roots, key=lambda r: abs(r - x0))
                    f_best = f(best_r)
                    if math.isfinite(f_best) and abs(f_best) <= TOL:
                        best_r, f_best = polish_root(best_r, f_best)
                        if is_finite(best_r) and abs(f_best) <= TOL:
                            return best_r
            except np.linalg.LinAlgError:
                pass
    except Exception:
        pass

    # Strategy 4: Safeguarded Newton
    x = x0
    fx = fx0

    for i in range(30):
        if not check_deadline():
            break

        if abs(fx) <= TOL:
            return x

        h = 1e-5 * max(1.0, abs(x))

        try:
            fx_h = f(x + h)
            fx_mh = f(x - h)
        except Exception:
            break

        if not (is_finite(fx_h) and is_finite(fx_mh)):
            break

        dfx = (fx_h - fx_mh) / (2 * h)

        if abs(dfx) < 1e-15:
            break

        step = fx / dfx
        x_new = x - step

        if not is_finite(x_new) or abs(step) > 10 * max(1.0, abs(x)):
            damp = 0.5
            improved = False
            for _ in range(8):
                x_damped = x - damp * step
                if is_finite(x_damped):
                    try:
                        fx_damped = f(x_damped)
                    except Exception:
                        damp *= 0.5
                        continue
                    if is_finite(fx_damped):
                        if abs(fx_damped) < best_f_abs:
                            best_f_abs = abs(fx_damped)
                            best_x = x_damped
                        if abs(fx_damped) <= TOL:
                            return x_damped
                        if abs(fx_damped) < abs(fx):
                            x = x_damped
                            fx = fx_damped
                            improved = True
                            break
                damp *= 0.5
            if not improved:
                break
            continue

        try:
            fx_new = f(x_new)
        except Exception:
            break

        if not is_finite(fx_new):
            break

        if abs(fx_new) < best_f_abs:
            best_f_abs = abs(fx_new)
            best_x = x_new

        if abs(fx_new) <= TOL:
            return x_new

        if abs(fx_new) > abs(fx) * 1.1 and i > 2:
            break

        x = x_new
        fx = fx_new

    # Final polish on best candidate
    if is_finite(best_x):
        try:
            f_best = f(best_x)
            if is_finite(f_best):
                best_x, f_best = polish_root(best_x, f_best)
                if is_finite(best_x) and abs(f_best) <= TOL:
                    return best_x
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
