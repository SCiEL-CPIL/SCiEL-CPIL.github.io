"""Root finding: Ill-conditioned - robust multi-strategy solver."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 using at most 500 evaluations."""
    start_time = time.time()
    WALL_LIMIT = 115.0  # safety margin before 120s
    MAX_EVALS = 500
    TOL = 1e-8

    eval_count = [0]

    def ef(x):
        eval_count[0] += 1
        try:
            v = f(x)
        except Exception:
            return 0.0
        if not math.isfinite(v):
            return 0.0
        return v

    def time_left():
        return (start_time + WALL_LIMIT) - time.time()

    def budget_ok():
        return eval_count[0] < MAX_EVALS and time_left() > 2.0

    def is_good(x, fx):
        return abs(fx) <= TOL

    x = float(x0)

    # --- Phase 1: adaptive Newton with progress monitoring ---
    best_x = x
    best_fx = ef(x)
    best_abs = abs(best_fx)
    if is_good(x, best_fx):
        return x

    h = 1e-7 * max(1.0, abs(x))
    prev_x = x
    stall = 0

    for i in range(120):
        if not budget_ok():
            break
        fx = best_fx
        if abs(fx) <= TOL:
            return x
        # finite-difference derivative
        d = (ef(x + h) - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        step = fx / d
        # adaptive damping
        if not math.isfinite(step):
            break
        x_new = x - step
        # clamp step size to avoid wild jumps
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            x_new = x - math.copysign(max_step, step)
        fx_new = ef(x_new)
        if not math.isfinite(fx_new):
            break
        if abs(fx_new) < best_abs:
            best_x = x_new
            best_abs = abs(fx_new)
            stall = 0
        else:
            stall += 1
            # backtrack / damp
            x_new = x - 0.5 * (x_new - x)
            fx_new = ef(x_new)
            if abs(fx_new) < best_abs:
                best_x = x_new
                best_abs = abs(fx_new)
                stall = 0
        x = x_new
        if stall > 5:
            break
        if abs(x - prev_x) < 1e-14 * max(1.0, abs(x)):
            break
        prev_x = x

    if is_good(best_x, ef(best_x)) if False else abs(best_fx) <= TOL:
        return best_x

    # --- Phase 2: secant method from two nearby points ---
    if budget_ok():
        a = x
        b = x + 1e-4 * max(1.0, abs(x))
        fa = best_fx
        fb = ef(b)
        for _ in range(80):
            if not budget_ok():
                break
            if abs(fb) < best_abs:
                best_x, best_abs = b, abs(fb)
            if abs(fa) < best_abs:
                best_x, best_abs = a, abs(fa)
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            if fb == fa:
                break
            c = b - fb * (b - a) / (fb - fa)
            if not math.isfinite(c):
                break
            fc = ef(c)
            a, fa = b, fb
            b, fb = c, fc
        if abs(fb) <= TOL:
            return b
        if abs(fa) <= TOL:
            return a
        if abs(fb) < best_abs:
            best_x, best_abs = b, abs(fb)
        if abs(fa) < best_abs:
            best_x, best_abs = a, abs(fa)

    # --- Phase 3: bisection if we can bracket a sign change ---
    if budget_ok():
        # try to find a bracket around best_x
        lo = best_x
        f_lo = ef(lo)
        hi = best_x
        f_hi = f_lo
        step = 1e-3 * max(1.0, abs(best_x))
        found = False
        for k in range(1, 60):
            if not budget_ok():
                break
            cand = best_x + k * step
            fc = ef(cand)
            if math.copysign(1.0, f_lo) != math.copysign(1.0, fc) and fc != 0:
                lo, f_lo = best_x, f_lo
                hi, f_hi = cand, fc
                found = True
                break
            cand2 = best_x - k * step
            fc2 = ef(cand2)
            if math.copysign(1.0, f_lo) != math.copysign(1.0, fc2) and fc2 != 0:
                lo, f_lo = cand2, fc2
                hi, f_hi = best_x, f_lo
                found = True
                break
            if abs(fc) < best_abs:
                best_x, best_abs = cand, abs(fc)
            if abs(fc2) < best_abs:
                best_x, best_abs = cand2, abs(fc2)
        if found:
            for _ in range(60):
                if not budget_ok():
                    break
                mid = 0.5 * (lo + hi)
                fm = ef(mid)
                if abs(fm) < best_abs:
                    best_x, best_abs = mid, abs(fm)
                if abs(fm) <= TOL:
                    return mid
                if math.copysign(1.0, fm) == math.copysign(1.0, f_lo):
                    lo, f_lo = mid, fm
                else:
                    hi, f_hi = mid, fm

    # --- Phase 4: final polish via Newton on best point ---
    if budget_ok():
        x = best_x
        fx = ef(x)
        h = 1e-8 * max(1.0, abs(x))
        for _ in range(30):
            if not budget_ok():
                break
            if abs(fx) <= TOL:
                return x
            d = (ef(x + h) - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = ef(x_new)
            if abs(fx_new) < best_abs:
                best_x, best_abs = x_new, abs(fx_new)
            x, fx = x_new, fx_new
            if abs(x_new - x) < 1e-15 * max(1.0, abs(x)):
                break

    return best_x

# EVOLVE-BLOCK-END
