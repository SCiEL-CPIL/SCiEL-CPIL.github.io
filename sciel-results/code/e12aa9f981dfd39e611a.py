"""Root finding: Ill-conditioned - robust solver with multiple strategies."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a portfolio approach: aggressive bracketing, bisection when bracketed,
    secant/Newton when not, with adaptive restarts.
    """
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]

    def eval_f(x):
        eval_count[0] += 1
        try:
            val = f(x)
        except (ValueError, OverflowError, ZeroDivisionError, ArithmeticError):
            return float('inf')
        if not math.isfinite(val):
            return float('inf')
        return val

    def check_time():
        return (time.time() - start_time) < MAX_WALL and eval_count[0] < MAX_EVALS

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def find_bracket_around(x, fx):
        """Try to find a bracket [a,b] containing x where f changes sign."""
        if not math.isfinite(fx):
            return None
        base_h = max(1e-6, abs(x) * 1e-6)
        # Try a sequence of scales, both symmetric and slightly asymmetric
        scales = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        for s in scales:
            h = s * max(1.0, abs(x))
            if h < 1e-15:
                continue
            a = x - h
            b = x + h
            fa = eval_f(a)
            fb = eval_f(b)
            if math.isfinite(fa) and math.isfinite(fb):
                if fa * fb < 0:
                    return (a, fa, b, fb)
                if abs(fa) <= TARGET_TOL:
                    return (a, fa, a, fa)
                if abs(fb) <= TARGET_TOL:
                    return (b, fb, b, fb)
            # Also try asymmetric: [x, x+h] and [x-h, x]
            if math.isfinite(fa) and math.isfinite(fx):
                if fa * fx < 0:
                    return (a, fa, x, fx)
            if math.isfinite(fb) and math.isfinite(fx):
                if fb * fx < 0:
                    return (x, fx, b, fb)
        return None

    def bisection(a, fa, b, fb, max_iters=60):
        """Bisection on [a,b] with sign change. Returns (x, fx) of best found."""
        if not math.isfinite(fa) or not math.isfinite(fb):
            return None, None
        if fa * fb > 0:
            return None, None
        if abs(fa) <= TARGET_TOL:
            return a, fa
        if abs(fb) <= TARGET_TOL:
            return b, fb

        for _ in range(max_iters):
            if not check_time():
                break
            mid = (a + b) / 2.0
            if mid == a or mid == b:
                break
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
        # Return the endpoint with smaller |f|
        if abs(fa) < abs(fb):
            return a, fa
        return b, fb

    def secant_step(x0, f0, x1, f1):
        """Secant step from (x0,f0) and (x1,f1). Returns (x_new, f_new) or (None, None)."""
        if not check_time():
            return None, None
        dx = x1 - x0
        if abs(dx) < 1e-15:
            return None, None
        df = (f1 - f0) / dx
        if abs(df) < 1e-15:
            return None, None
        step = f1 / df
        max_step = 5.0 * max(1.0, abs(x1))
        if abs(step) > max_step:
            step = max_step * math.copysign(1.0, step)
        x_new = x1 - step
        f_new = eval_f(x_new)
        if not math.isfinite(f_new):
            return None, None
        return x_new, f_new

    def newton_step(x, fx):
        """Damped Newton step with FD derivative. Returns (x_new, f_new) or (None, None)."""
        if not check_time():
            return None, None
        # Central difference for better accuracy
        h = 1e-7 * max(1.0, abs(x))
        fp = eval_f(x + h)
        fm = eval_f(x - h)
        if math.isfinite(fp) and math.isfinite(fm):
            df = (fp - fm) / (2 * h)
        elif math.isfinite(fp):
            df = (fp - fx) / h
        elif math.isfinite(fm):
            df = (fx - fm) / h
        else:
            return None, None
        if abs(df) < 1e-15:
            return None, None
        step = fx / df
        max_step = 5.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step * math.copysign(1.0, step)
        x_new = x - step
        f_new = eval_f(x_new)
        if not math.isfinite(f_new):
            # Try halving the step
            step *= 0.5
            x_new = x - step
            f_new = eval_f(x_new)
            if not math.isfinite(f_new):
                return None, None
        return x_new, f_new

    # === Main algorithm ===
    x = float(x0)
    fx = eval_f(x)
    best_x = x
    best_fx = fx

    if is_good(x, fx):
        return x

    # Track two most recent points for secant
    x_hist = [(x, fx)]

    # Phase 1: Try to find a bracket
    bracket = find_bracket_around(x, fx)
    if bracket is not None:
        a, fa, b, fb = bracket
        rx, rf = bisection(a, fa, b, fb)
        if rx is not None and rf is not None:
            if is_good(rx, rf):
                return rx
            if abs(rf) < abs(best_fx):
                best_x, best_fx = rx, rf

    # Phase 2: Iterative refinement with secant/Newton
    x_curr = best_x
    fx_curr = best_fx
    x_prev = x
    fx_prev = fx

    for iteration in range(300):
        if not check_time():
            break
        if is_good(x_curr, fx_curr):
            return x_curr

        # Try to find a bracket around current point
        bracket = find_bracket_around(x_curr, fx_curr)
        if bracket is not None:
            a, fa, b, fb = bracket
            rx, rf = bisection(a, fa, b, fb)
            if rx is not None and rf is not None:
                if is_good(rx, rf):
                    return rx
                if abs(rf) < abs(fx_curr):
                    x_prev, fx_prev = x_curr, fx_curr
                    x_curr, fx_curr = rx, rf
                    if abs(fx_curr) < abs(best_fx):
                        best_x, best_fx = x_curr, fx_curr
                    continue

        # Try secant step
        if len(x_hist) >= 2:
            x0_s, f0_s = x_hist[-2]
            x1_s, f1_s = x_hist[-1]
            if abs(x1_s - x0_s) > 1e-15:
                sn_x, sn_f = secant_step(x0_s, f0_s, x1_s, f1_s)
                if sn_x is not None and sn_f is not None:
                    if is_good(sn_x, sn_f):
                        return sn_x
                    if abs(sn_f) < abs(fx_curr) * 0.95:
                        x_prev, fx_prev = x_curr, fx_curr
                        x_curr, fx_curr = sn_x, sn_f
                        if abs(fx_curr) < abs(best_fx):
                            best_x, best_fx = x_curr, fx_curr
                        x_hist.append((x_curr, fx_curr))
                        if len(x_hist) > 5:
                            x_hist.pop(0)
                        continue

        # Try Newton step
        nt_x, nt_f = newton_step(x_curr, fx_curr)
        if nt_x is not None and nt_f is not None:
            if is_good(nt_x, nt_f):
                return nt_x
            if abs(nt_f) < abs(fx_curr) * 0.95:
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = nt_x, nt_f
                if abs(fx_curr) < abs(best_fx):
                    best_x, best_fx = x_curr, fx_curr
                x_hist.append((x_curr, fx_curr))
                if len(x_hist) > 5:
                    x_hist.pop(0)
                continue

        # Stuck: try restarts with perturbations
        perturb_scales = [1e-3, 1e-2, 0.1, 1.0, 10.0]
        found = False
        for ps in perturb_scales:
            if not check_time():
                break
            for sign in [1, -1]:
                if not check_time():
                    break
                x_pert = x_curr + sign * ps * max(1.0, abs(x_curr))
                fx_pert = eval_f(x_pert)
                if math.isfinite(fx_pert) and abs(fx_pert) < abs(fx_curr) * 0.9:
                    x_prev, fx_prev = x_curr, fx_curr
                    x_curr, fx_curr = x_pert, fx_pert
                    if abs(fx_curr) < abs(best_fx):
                        best_x, best_fx = x_curr, fx_curr
                    x_hist.append((x_curr, fx_curr))
                    if len(x_hist) > 5:
                        x_hist.pop(0)
                    found = True
                    break
            if found:
                break
        if not found:
            # Try a larger jump
            x_jump = x_curr + 0.5 * max(1.0, abs(x_curr))
            fx_jump = eval_f(x_jump)
            if math.isfinite(fx_jump) and abs(fx_jump) < abs(fx_curr):
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = x_jump, fx_jump
                if abs(fx_curr) < abs(best_fx):
                    best_x, best_fx = x_curr, fx_curr
                x_hist.append((x_curr, fx_curr))
                if len(x_hist) > 5:
                    x_hist.pop(0)

    # Final polish: try to tighten the best solution
    if check_time() and abs(best_fx) > TARGET_TOL:
        x_pol = best_x
        fx_pol = best_fx
        for _ in range(30):
            if not check_time():
                break
            if abs(fx_pol) <= 1e-10:
                break
            h = 1e-8 * max(1.0, abs(x_pol))
            fp = eval_f(x_pol + h)
            fm = eval_f(x_pol - h)
            if math.isfinite(fp) and math.isfinite(fm):
                df = (fp - fm) / (2 * h)
            elif math.isfinite(fp):
                df = (fp - fx_pol) / h
            elif math.isfinite(fm):
                df = (fx_pol - fm) / h
            else:
                break
            if abs(df) < 1e-15:
                break
            step = fx_pol / df
            max_step = 1.0 * max(1.0, abs(x_pol))
            if abs(step) > max_step:
                step = max_step * math.copysign(1.0, step)
            x_new = x_pol - step
            fx_new = eval_f(x_new)
            if math.isfinite(fx_new) and abs(fx_new) < abs(fx_pol):
                x_pol, fx_pol = x_new, fx_new
            else:
                # Try half step
                x_new = x_pol - step * 0.5
                fx_new = eval_f(x_new)
                if math.isfinite(fx_new) and abs(fx_new) < abs(fx_pol):
                    x_pol, fx_pol = x_new, fx_new
                else:
                    break
        if abs(fx_pol) < abs(best_fx):
            best_x, best_fx = x_pol, fx_pol

    return best_x

# EVOLVE-BLOCK-END
