"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under 120s

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

    # Try several initial step sizes to find a good second point
    step_factors = [1e-4, 1e-3, 1e-2, 0.1, 1.0, -1e-4, -1e-3, -1e-2, -0.1, -1.0]
    x2 = None
    fx2 = None

    for sf in step_factors:
        h = sf * max(1.0, abs(x))
        x2 = x + h
        try:
            fx2 = f(x2)
        except Exception:
            continue
        call_count += 1
        if not math.isfinite(fx2):
            continue
        if abs(fx2) <= 1e-8:
            return x2
        if abs(fx2) < abs(best_fx):
            best_x = x2
            best_fx = fx2
        # Accept this point if it gives a reasonable slope estimate
        if abs(fx2 - fx) > 1e-15 * max(1.0, abs(fx), abs(fx2)):
            break
        x2 = None
        fx2 = None

    if x2 is None or fx2 is None:
        # All attempts failed; return best
        return best_x

    # If we used more than 2 calls for initialization, reduce max iterations
    max_iter = 498 - call_count
    if max_iter < 10:
        max_iter = 10

    # Secant method iteration with stall detection
    prev_best = abs(best_fx)
    stall_count = 0

    for i in range(max_iter):
        if time.time() > deadline:
            return best_x

        denom = fx2 - fx
        if abs(denom) < 1e-30:
            # Nearly parallel; perturb
            h2 = 1e-2 * max(1.0, abs(x2))
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
            stall_count = 0
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
            stall_count = 0
        else:
            stall_count += 1

        if abs(fx_new) <= 1e-8:
            return x_new

        # If we've stalled (no improvement in last 5 steps), try a larger perturbation
        if stall_count >= 5:
            # Try stepping from best_x with a larger step
            h3 = 0.1 * max(1.0, abs(best_x)) * (1.0 if (i % 2 == 0) else -1.0)
            x_pert = best_x + h3
            try:
                fx_pert = f(x_pert)
            except Exception:
                return best_x
            call_count += 1
            if math.isfinite(fx_pert):
                if abs(fx_pert) < abs(best_fx):
                    best_x = x_pert
                    best_fx = fx_pert
                if abs(fx_pert) <= 1e-8:
                    return x_pert
                # Use perturbed point as new x2
                x, fx = x2, fx2
                x2, fx2 = x_pert, fx_pert
            stall_count = 0
            continue

        # Shift: x = x2, x2 = x_new
        x, fx = x2, fx2
        x2, fx2 = x_new, fx_new

    return best_x
# EVOLVE-BLOCK-END
