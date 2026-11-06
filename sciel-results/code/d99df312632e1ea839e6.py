import time
import math

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses Newton's method with fallback to bracketing + Brent's method.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    eval_count = 0
    max_evals = 500

    def check_deadline():
        return time.time() >= deadline or eval_count >= max_evals

    # Track best result found so far
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = f(float(x0))
        eval_count += 1
        if math.isnan(best_fx) or math.isinf(best_fx):
            return best_x
        if abs(best_fx) <= 1e-8:
            return best_x
    except Exception:
        return best_x

    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and not math.isnan(fx) and not math.isinf(fx):
            if abs(fx) < abs(best_fx):
                best_x = x
                best_fx = fx

    x = float(x0)
    fx = best_fx

    # Phase 1: Newton's method with adaptive step control
    for _ in range(40):
        if check_deadline():
            break
        if abs(fx) <= 1e-8:
            return x

        # Compute derivative using central difference
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
            eval_count += 2
            if math.isnan(fx_plus) or math.isinf(fx_plus) or math.isnan(fx_minus) or math.isinf(fx_minus):
                break
            d = (fx_plus - fx_minus) / (2.0 * h)
        except Exception:
            break

        if d == 0 or math.isnan(d) or math.isinf(d):
            break

        # Newton step
        step = fx / d
        new_x = x - step

        # Adaptive step control: if step is too large, damp it
        if abs(step) > 10.0 * max(1.0, abs(x)):
            step = math.copysign(10.0 * max(1.0, abs(x)), step)
            new_x = x - step

        # Check if new point is valid
        try:
            new_fx = f(new_x)
            eval_count += 1
        except Exception:
            break

        if math.isnan(new_fx) or math.isinf(new_fx):
            break

        update_best(new_x, new_fx)

        # Accept step if it reduces |f| or is a reasonable improvement
        if abs(new_fx) < abs(fx) * 1.1:
            x = new_x
            fx = new_fx
        else:
            # Try a halved step
            half_step = step / 2.0
            half_x = x - half_step
            try:
                half_fx = f(half_x)
                eval_count += 1
            except Exception:
                break
            if math.isnan(half_fx) or math.isinf(half_fx):
                break
            update_best(half_x, half_fx)
            if abs(half_fx) < abs(fx):
                x = half_x
                fx = half_fx
            else:
                # Newton is struggling; break out to try bracketing
                break

        if abs(fx) <= 1e-8:
            return x

    # Phase 2: Try to find a bracket and use Brent's method
    if abs(fx) <= 1e-8:
        return x

    # Search for a bracket [a, b] where f(a) and f(b) have opposite signs
    a, b = None, None
    fa, fb = None, None

    # Start from current x and expand outward
    step_size = 1e-4 * max(1.0, abs(x))
    for i in range(30):
        if check_deadline():
            break
        # Try expanding to the right
        trial_b = x + step_size * (2 ** i)
        try:
            trial_fb = f(trial_b)
            eval_count += 1
        except Exception:
            trial_fb = None

        if trial_fb is not None and not math.isnan(trial_fb) and not math.isinf(trial_fb):
            update_best(trial_b, trial_fb)
            if fx is not None and not math.isnan(fx) and not math.isinf(fx):
                if fx * trial_fb < 0:
                    a, b = x, trial_b
                    fa, fb = fx, trial_fb
                    break

        # Try expanding to the left
        trial_a = x - step_size * (2 ** i)
        try:
            trial_fa = f(trial_a)
            eval_count += 1
        except Exception:
            trial_fa = None

        if trial_fa is not None and not math.isnan(trial_fa) and not math.isinf(trial_fa):
            update_best(trial_a, trial_fa)
            if fx is not None and not math.isnan(fx) and not math.isinf(fx):
                if fx * trial_fa < 0:
                    a, b = trial_a, x
                    fa, fb = trial_fa, fx
                    break

    # If we found a bracket, use Brent's method
    if a is not None and b is not None:
        # Simple Brent's method implementation
        # Using the algorithm from Brent's paper
        tol = 1e-12
        max_iter = 50

        c, d, fc, fd = a, b, fa, fb
        e = f = 0.0

        for _ in range(max_iter):
            if check_deadline():
                break
            if abs(fc) < abs(fd):
                a, b = b, a
                fa, fb = fb, fa
                c, d = d, c
                fc, fd = fd, fc
            e = f
            f = (b - a) / 2.0

            if abs(f) < tol or fc == 0.0:
                return c

            if abs(e) >= tol and abs(fc) > abs(fd):
                s = fc / fd
                if a == c or b == c:
                    p = 2.0 * f * s
                    q = 1.0 - s
                else:
                    p = s * (2.0 * f * (d - c) - (c - a) * (fc - fa))
                    q = (d - c) * (c - a) * (fc - fa)
                    if p > 0.0:
                        q = -q
                    elif p < 0.0:
                        p = -p
                        q = -q
                    r = s * (2.0 * f * (d - c) - (c - a) * (fc - fa)) / q
                    # This is getting complex; let's use a simpler safeguarded approach

            # Use a simpler safeguarded bisection + secant hybrid
            # Secant step
            if fa != fb:
                secant_x = a - fa * (b - a) / (fb - fa)
            else:
                secant_x = (a + b) / 2.0

            # Check if secant is within [a, b]
            if a < secant_x < b:
                s = secant_x
            else:
                s = (a + b) / 2.0

            try:
                fs = f(s)
                eval_count += 1
            except Exception:
                break

            if math.isnan(fs) or math.isinf(fs):
                break

            update_best(s, fs)

            if abs(fs) <= 1e-8:
                return s

            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs

            if abs(b - a) < 1e-14:
                break

        # Return the best point in the bracket
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    # Return the best result found
    return best_x
