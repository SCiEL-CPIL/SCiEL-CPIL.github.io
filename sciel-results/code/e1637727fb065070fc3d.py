"""Root finding: Ill-conditioned - robust hybrid solver."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach: Newton with damping, bisection fallback, secant method,
    and multiple restart strategies for ill-conditioned problems.
    """
    start_time = time.time()
    MAX_WALL = 110.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    # Global evaluation counter
    eval_count = [0]

    def eval_f(x):
        """Counted evaluation of f with finite check."""
        eval_count[0] += 1
        try:
            val = f(x)
        except (ValueError, OverflowError, ZeroDivisionError):
            return float('inf')
        if not math.isfinite(val):
            return float('inf')
        return val

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def check_time():
        return (time.time() - start_time) < MAX_WALL and eval_count[0] < MAX_EVALS

    def newton_step(x, fx, use_fd=True):
        """Take a damped Newton step. Returns (new_x, new_fx, success)."""
        if not check_time():
            return x, fx, False

        # Estimate derivative
        if use_fd:
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if not math.isfinite(fx_plus):
                h = 1e-5 * max(1.0, abs(x))
                fx_plus = eval_f(x + h)
                if not math.isfinite(fx_plus):
                    return x, fx, False
            df = (fx_plus - fx) / h
        else:
            # Use secant-like estimate from previous point if available
            return x, fx, False

        if abs(df) < 1e-15:
            return x, fx, False

        # Damped Newton step
        step = fx / df
        # Limit step size to avoid divergence
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step * math.copysign(1.0, step)

        x_new = x - step
        fx_new = eval_f(x_new)

        if not math.isfinite(fx_new):
            # Try smaller step
            step = step * 0.5
            x_new = x - step
            fx_new = eval_f(x_new)
            if not math.isfinite(fx_new):
                return x, fx, False

        return x_new, fx_new, True

    def secant_step(x_prev, fx_prev, x_curr, fx_curr):
        """Take a secant step. Returns (new_x, new_fx, success)."""
        if not check_time():
            return x_curr, fx_curr, False

        dx = x_curr - x_prev
        if abs(dx) < 1e-15:
            return x_curr, fx_curr, False

        df = (fx_curr - fx_prev) / dx
        if abs(df) < 1e-15:
            return x_curr, fx_curr, False

        step = fx_curr / df
        max_step = 10.0 * max(1.0, abs(x_curr))
        if abs(step) > max_step:
            step = max_step * math.copysign(1.0, step)

        x_new = x_curr - step
        fx_new = eval_f(x_new)

        if not math.isfinite(fx_new):
            return x_curr, fx_curr, False

        return x_new, fx_new, True

    def bisection(a, fa, b, fb, max_iters=50):
        """Bisection method on [a, b] where fa and fb have opposite signs."""
        if not math.isfinite(fa) or not math.isfinite(fb):
            return None, None
        if fa * fb > 0:
            return None, None

        for _ in range(max_iters):
            if not check_time():
                break
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            if not math.isfinite(fm):
                break
            if abs(fm) <= TARGET_TOL:
                return mid, fm
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                break
        return (a + b) / 2.0, eval_f((a + b) / 2.0)

    def find_bracket(x, fx):
        """Try to find a bracket around x."""
        h = 1e-3 * max(1.0, abs(x))
        for scale in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            h = scale * max(1.0, abs(x))
            x_left = x - h
            x_right = x + h
            fl = eval_f(x_left)
            fr = eval_f(x_right)
            if math.isfinite(fl) and math.isfinite(fr) and fl * fr < 0:
                return x_left, fl, x_right, fr
        return None

    # Main algorithm
    x = float(x0)
    fx = eval_f(x)
    best_x = x
    best_fx = fx

    if is_good(x, fx):
        return x

    # Track history for secant
    x_prev = x
    fx_prev = fx

    for iteration in range(200):
        if not check_time():
            break

        # Check if current is good enough
        if is_good(x, fx):
            best_x = x
            best_fx = fx
            break

        # Try to find a bracket for bisection
        bracket = find_bracket(x, fx)
        if bracket is not None:
            a, fa, b, fb = bracket
            # If we have a bracket, use bisection
            result_x, result_fx = bisection(a, fa, b, fb)
            if result_x is not None and result_fx is not None:
                if is_good(result_x, result_fx):
                    return result_x
                if abs(result_fx) < abs(fx):
                    x, fx = result_x, result_fx

        # Try Newton step
        x_new, fx_new, success = newton_step(x, fx)
        if success and math.isfinite(fx_new):
            if is_good(x_new, fx_new):
                return x_new
            if abs(fx_new) < abs(fx) * 1.01:
                # Accept the step
                x_prev, fx_prev = x, fx
                x, fx = x_new, fx_new
                if abs(fx) < abs(best_fx):
                    best_x, best_fx = x, fx
            else:
                # Newton step didn't help, try secant
                x_new2, fx_new2, success2 = secant_step(x_prev, fx_prev, x, fx)
                if success2 and math.isfinite(fx_new2):
                    if is_good(x_new2, fx_new2):
                        return x_new2
                    if abs(fx_new2) < abs(fx) * 1.01:
                        x_prev, fx_prev = x, fx
                        x, fx = x_new2, fx_new2
                        if abs(fx) < abs(best_fx):
                            best_x, best_fx = x, fx
                    else:
                        # Try smaller Newton step
                        h = 1e-7 * max(1.0, abs(x))
                        fx_plus = eval_f(x + h)
                        if math.isfinite(fx_plus):
                            df = (fx_plus - fx) / h
                            if abs(df) > 1e-15:
                                step = fx / df
                                # Damp the step
                                for damp in [0.5, 0.25, 0.1, 0.05, 0.01]:
                                    x_damp = x - damp * step
                                    fx_damp = eval_f(x_damp)
                                    if math.isfinite(fx_damp) and abs(fx_damp) < abs(fx):
                                        x_prev, fx_prev = x, fx
                                        x, fx = x_damp, fx_damp
                                        if abs(fx) < abs(best_fx):
                                            best_x, best_fx = x, fx
                                        break
        else:
            # Newton failed, try secant
            x_new2, fx_new2, success2 = secant_step(x_prev, fx_prev, x, fx)
            if success2 and math.isfinite(fx_new2):
                if is_good(x_new2, fx_new2):
                    return x_new2
                if abs(fx_new2) < abs(fx) * 1.01:
                    x_prev, fx_prev = x, fx
                    x, fx = x_new2, fx_new2
                    if abs(fx) < abs(best_fx):
                        best_x, best_fx = x, fx

        # If we're stuck, try a restart with a small perturbation
        if iteration > 30 and abs(fx) >= abs(best_fx) * 0.9:
            # Perturb x slightly
            perturbation = 1e-4 * max(1.0, abs(x)) * (1 if iteration % 2 == 0 else -1)
            x_pert = x + perturbation
            fx_pert = eval_f(x_pert)
            if math.isfinite(fx_pert) and abs(fx_pert) < abs(fx):
                x_prev, fx_prev = x, fx
                x, fx = x_pert, fx_pert

    # Final polish: try to improve best_x with tighter tolerance
    if check_time():
        x_curr = best_x
        fx_curr = best_fx
        for _ in range(20):
            if not check_time():
                break
            if abs(fx_curr) <= 1e-10:
                break
            h = 1e-8 * max(1.0, abs(x_curr))
            fx_plus = eval_f(x_curr + h)
            if math.isfinite(fx_plus):
                df = (fx_plus - fx_curr) / h
                if abs(df) > 1e-15:
                    step = fx_curr / df
                    max_step = 1.0 * max(1.0, abs(x_curr))
                    if abs(step) > max_step:
                        step = max_step * math.copysign(1.0, step)
                    x_new = x_curr - step
                    fx_new = eval_f(x_new)
                    if math.isfinite(fx_new) and abs(fx_new) < abs(fx_curr):
                        x_curr, fx_curr = x_new, fx_new
                    else:
                        break
            else:
                break
        if abs(fx_curr) < abs(best_fx):
            best_x, best_fx = x_curr, fx_curr

    return best_x

# EVOLVE-BLOCK-END
