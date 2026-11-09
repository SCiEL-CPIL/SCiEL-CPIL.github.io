import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Strategy:
    1. Evaluate f(x0). If good, return.
    2. Try a few small perturbations to find a second point (or a better first point).
    3. Use Secant method for fast local convergence (superlinear).
    4. If Secant fails to converge quickly or diverges, switch to a bracketing
       hybrid (Bisection + Inverse Quadratic Interpolation) to guarantee convergence
       if a sign change is found.
    5. If no bracket is found, return the best point encountered.
    """
    start_time = time.time()
    deadline = start_time + 8.0

    tol_final = 1e-8

    x0 = float(x0)

    # --- Phase 0: Initial Evaluation and Perturbation ---
    f0 = f(x0)
    if not math.isfinite(f0):
        best_x = x0
        best_f = float('inf')
        base_step = 1e-4 * max(1.0, abs(x0))
        found_finite = False
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512, 1024, -1024]:
            if time.time() > deadline:
                return best_x if math.isfinite(best_f) else x0
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try):
                if not found_finite:
                    x0, f0 = x_try, f_try
                    found_finite = True
                    break
                if abs(f_try) < abs(best_f):
                    best_x, best_f = x_try, f_try
            if abs(f_try) <= tol_final:
                return x_try
        if not found_finite:
            return x0

    if abs(f0) <= tol_final:
        return x0

    # We need a second point for Secant.
    base_step = 1e-5 * max(1.0, abs(x0))

    x_prev = x0
    f_prev = f0
    x_curr = x0 + base_step
    f_curr = f(x_curr)

    if abs(f_curr) <= tol_final:
        return x_curr

    if not math.isfinite(f_curr):
        x_curr = x0 - base_step
        f_curr = f(x_curr)
        if not math.isfinite(f_curr):
            x_curr = x0 + base_step * 10
            f_curr = f(x_curr)
            if not math.isfinite(f_curr):
                return x0

    best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
    best_f = min(abs(f_prev), abs(f_curr))

    # --- Phase 1: Secant Method ---
    max_secant_iter = 25

    for _ in range(max_secant_iter):
        if time.time() > deadline:
            break

        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            break

        x_next = x_curr - f_curr * (x_curr - x_prev) / denom

        if not math.isfinite(x_next):
            break

        f_next = f(x_next)

        if not math.isfinite(f_next):
            break

        if abs(f_next) < best_f:
            best_x = x_next
            best_f = abs(f_next)

        if abs(f_next) <= tol_final:
            return x_next

        if abs(x_next) > 1e10 * max(1.0, abs(x0)):
            break

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    if abs(f_curr) <= tol_final:
        return x_curr
    if abs(f_prev) <= tol_final:
        return x_prev

    # --- Phase 2: Bracketing Search ---
    a, fa = x_prev, f_prev
    b, fb = x_curr, f_curr

    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    if fa * fb < 0:
        bracket_found = True
    else:
        bracket_found = False
        if abs(fa) < abs(fb):
            step_size = max(abs(b - a), 1e-4 * max(1.0, abs(a)))

            for i in range(10):
                if time.time() > deadline:
                    break
                b_new = b + step_size * (1 + i)
                fb_new = f(b_new)
                if math.isfinite(fb_new):
                    if fa * fb_new < 0:
                        b, fb = b_new, fb_new
                        bracket_found = True
                        break
                a_new = a - step_size * (1 + i)
                fa_new = f(a_new)
                if math.isfinite(fa_new):
                    if fa_new * fb < 0:
                        a, fa = a_new, fa_new
                        bracket_found = True
                        break

            if not bracket_found:
                wide_step = 1.0 * max(1.0, abs(a))
                for i in range(5):
                    if time.time() > deadline:
                        break
                    x_try = a + wide_step * (i + 1)
                    f_try = f(x_try)
                    if math.isfinite(f_try):
                        if fa * f_try < 0:
                            b, fb = x_try, f_try
                            bracket_found = True
                            break
                    x_try = a - wide_step * (i + 1)
                    f_try = f(x_try)
                    if math.isfinite(f_try):
                        if fa * f_try < 0:
                            a, fa = x_try, f_try
                            bracket_found = True
                            break
                    if abs(f_try) <= tol_final:
                        return x_try

    if not bracket_found:
        return best_x

    # --- Phase 3: Hybrid Bisection / Inverse Quadratic Interpolation ---
    m = (a + b) / 2.0
    fm = f(m)

    if not math.isfinite(fm):
        for _ in range(100):
            if time.time() > deadline:
                break
            if abs(fa) <= tol_final:
                return a
            if abs(fb) <= tol_final:
                return b
            if abs(b - a) < 1e-15:
                return (a + b) / 2.0
            mid = (a + b) / 2.0
            fmid = f(mid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= tol_final:
                return mid
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
        return a if abs(fa) < abs(fb) else b

    if abs(fm) <= tol_final:
        return m

    points = [(a, fa), (m, fm), (b, fb)]
    points.sort(key=lambda p: p[0])
    x1, f1 = points[0]
    x2, f2 = points[1]
    x3, f3 = points[2]

    if f1 * f2 < 0:
        x_lo, f_lo, x_hi, f_hi = x1, f1, x2, f2
        x_mid, f_mid = x3, f3
    elif f2 * f3 < 0:
        x_lo, f_lo, x_hi, f_hi = x2, f2, x3, f3
        x_mid, f_mid = x1, f1
    else:
        return best_x

    max_hybrid_iter = 100
    for _ in range(max_hybrid_iter):
        if time.time() > deadline:
            break

        if abs(f_lo) <= tol_final:
            return x_lo
        if abs(f_hi) <= tol_final:
            return x_hi
        if abs(f_mid) <= tol_final:
            return x_mid

        if abs(x_hi - x_lo) < 1e-15:
            return (x_lo + x_hi) / 2.0

        d1 = (f_lo - f_mid) * (f_lo - f_hi)
        d2 = (f_mid - f_lo) * (f_mid - f_hi)
        d3 = (f_hi - f_lo) * (f_hi - f_mid)

        x_iqi = None
        if abs(d1) > 1e-30 and abs(d2) > 1e-30 and abs(d3) > 1e-30:
            x_iqi = (x_lo * f_mid * f_hi / d1 +
                     x_mid * f_lo * f_hi / d2 +
                     x_hi * f_lo * f_mid / d3)

        margin = 1e-12 * max(1.0, abs(x_hi - x_lo))
        if (x_iqi is not None and math.isfinite(x_iqi) and
            (x_lo + margin < x_iqi < x_hi - margin)):
            x_new = x_iqi
        else:
            x_new = (x_lo + x_hi) / 2.0

        f_new = f(x_new)

        if not math.isfinite(f_new):
            x_new = (x_lo + x_hi) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break

        if abs(f_new) <= tol_final:
            return x_new

        if f_lo * f_new < 0:
            x_hi, f_hi = x_new, f_new
        else:
            x_lo, f_lo = x_new, f_new

        if abs(x_hi - x_lo) < 1e-10:
            candidates = [(x_lo, f_lo), (x_hi, f_hi), (x_mid, f_mid)]
            best_local = min(candidates, key=lambda p: abs(p[1]))
            if abs(best_local[1]) <= tol_final:
                return best_local[0]
            break

    candidates = [(x_lo, f_lo), (x_hi, f_hi), (x_mid, f_mid), (best_x, best_f)]
    best_final = min(candidates, key=lambda p: abs(p[1]))
    return best_final[0]

# EVOLVE-BLOCK-END
