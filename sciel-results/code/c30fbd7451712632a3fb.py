"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    x0 = float(x0)
    eval_count = 0
    max_evals = 500

    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        eval_count += 1
        try:
            val = f(x)
            if not np.isfinite(val):
                return None
            return float(val)
        except:
            return None

    # Initialize
    fx0 = safe_f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_f = fx0
    best_abs_f = abs(fx0)

    # Phase 1: Robust Secant Method with Damping
    # Use two points to start secant
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = safe_f(x1)

    if fx1 is None:
        x1 = x0 - h
        fx1 = safe_f(x1)

    if fx1 is None:
        x_m = x0 - h
        x_p = x0 + h
        fm = safe_f(x_m)
        fp = safe_f(x_p)
        if fm is not None and fp is not None:
            d = (fp - fm) / (2 * h)
            if d != 0:
                x1 = x0 - fx0 / d
                fx1 = safe_f(x1)
            else:
                fx1 = None
        if fx1 is None:
            return best_x

    # Track history for secant
    x_prev = x0
    fx_prev = fx0
    x_curr = x1
    fx_curr = fx1

    for i in range(80):
        if time.time() > deadline:
            break
        if fx_curr is None:
            break
        if abs(fx_curr) < best_abs_f:
            best_x = x_curr
            best_f = fx_curr
            best_abs_f = abs(fx_curr)
        if abs(fx_curr) <= 1e-8:
            return x_curr

        # Secant step
        if fx_curr == fx_prev:
            x_curr = x_curr + 1e-4
            fx_curr = safe_f(x_curr)
            continue

        x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)

        # Damping
        step = x_next - x_curr
        max_step = 0.5 * abs(x_curr) + 1.0
        if abs(step) > max_step:
            x_next = x_curr + max_step * math.copysign(1, step)

        fx_next = safe_f(x_next)

        if fx_next is None:
            x_next = x_curr + 0.5 * step
            fx_next = safe_f(x_next)
            if fx_next is None:
                break

        if abs(fx_next) < abs(fx_curr):
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_next
            fx_curr = fx_next
        else:
            x_damped = x_curr + 0.5 * (x_next - x_curr)
            fx_damped = safe_f(x_damped)
            if fx_damped is not None and abs(fx_damped) < abs(fx_curr):
                x_prev = x_curr
                fx_prev = fx_curr
                x_curr = x_damped
                fx_curr = fx_damped
            else:
                break

    if abs(best_f) <= 1e-8:
        return best_x

    # Phase 2: Bracketing
    a = best_x
    fa = safe_f(a)
    if fa is None:
        return best_x

    r = 1e-4 * max(1.0, abs(a))
    found_bracket = False
    b = a
    fb = None

    for i in range(30):
        if time.time() > deadline:
            break
        if eval_count >= max_evals - 10:
            break

        b = a + r
        fb = safe_f(b)
        if fb is not None:
            if fa * fb < 0:
                found_bracket = True
                break
            if abs(fb) < best_abs_f:
                best_x = b
                best_f = fb
                best_abs_f = abs(fb)

        b = a - r
        fb = safe_f(b)
        if fb is not None:
            if fa * fb < 0:
                found_bracket = True
                break
            if abs(fb) < best_abs_f:
                best_x = b
                best_f = fb
                best_abs_f = abs(fb)

        r *= 2.0

    if found_bracket:
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if root is not None:
                fr = safe_f(root)
                if fr is not None and abs(fr) < best_abs_f:
                    return root
        except:
            pass

    if abs(best_f) <= 1e-8:
        return best_x

    # Phase 3: Newton's method with central difference derivative
    # This is more robust than forward difference for ill-conditioned problems
    if time.time() < deadline and eval_count < max_evals - 20:
        x = best_x
        for i in range(50):
            if time.time() > deadline or eval_count >= max_evals - 5:
                break
            fx = safe_f(x)
            if fx is None:
                break
            if abs(fx) < best_abs_f:
                best_x = x
                best_f = fx
                best_abs_f = abs(fx)
            if abs(fx) <= 1e-8:
                return x

            h = 1e-6 * max(1.0, abs(x))
            fp = safe_f(x + h)
            fm = safe_f(x - h)
            if fp is None or fm is None:
                break
            d = (fp - fm) / (2 * h)
            if d == 0 or not np.isfinite(d):
                break
            step = fx / d
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            x_new = x - step
            if not np.isfinite(x_new):
                break
            fx_new = safe_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) < abs(fx) * 0.5 or abs(fx_new) <= 1e-8:
                x = x_new
            else:
                x = x - step * 0.5
                if not np.isfinite(x):
                    break

    return best_x

# EVOLVE-BLOCK-END
