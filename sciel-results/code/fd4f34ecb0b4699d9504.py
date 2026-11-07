"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under 120s

    x = float(x0)

    # Get first evaluation
    try:
        fx = f(x)
    except Exception:
        return x

    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_fx = fx
    call_count = 1

    # Get second point with adaptive step
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    try:
        fx2 = f(x2)
    except Exception:
        return best_x

    call_count += 1
    if not math.isfinite(fx2):
        return best_x
    if abs(fx2) < abs(best_fx):
        best_x = x2
        best_fx = fx2
    if abs(fx2) <= 1e-8:
        return x2

    # Secant method iteration
    for i in range(480):  # 2 + 480 = 482 calls max, well under 500
        if time.time() > deadline:
            return best_x

        denom = fx2 - fx
        if abs(denom) < 1e-30:
            # Nearly parallel; try a different step from x2
            h2 = 1e-3 * max(1.0, abs(x2))
            x_new = x2 + h2
            try:
                fx_new = f(x_new)
            except Exception:
                return best_x
            call_count += 1
            if not math.isfinite(fx_new):
                return best_x
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            x, fx = x2, fx2
            x2, fx2 = x_new, fx_new
            continue

        x_new = x2 - fx2 * (x2 - x) / denom
        if not math.isfinite(x_new):
            return best_x

        try:
            fx_new = f(x_new)
        except Exception:
            return best_x

        call_count += 1
        if not math.isfinite(fx_new):
            return best_x

        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        if abs(fx_new) <= 1e-8:
            return x_new

        # Shift: x = x2, x2 = x_new
        x, fx = x2, fx2
        x2, fx2 = x_new, fx_new

    return best_x
# EVOLVE-BLOCK-END
