import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 108.0
    max_calls = 420
    call_count = 0

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    if not math.isfinite(x):
        return x

    fx = f_call(x)

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_f = fx

    # Phase 1: Try to find a bracket using geometric expansion
    # If we find one, use bisection (guaranteed convergence)
    # If we don't find one within a reasonable number of calls,
    # fall back to a derivative-free optimization approach

    # First, try to find a bracket
    a = x
    fa = fx
    b = None
    fb = None
    bracket_found = False

    # Try expanding in both directions
    for direction in [1, -1]:
        if not time_ok():
            break
        a = x
        fa = fx
        h = 0.5
        for _ in range(20):
            if not time_ok():
                break
            b = a + direction * h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if abs(fb) < abs(best_f):
                best_x = b
                best_f = fb
            if fa * fb < 0:
                bracket_found = True
                break
            a = b
            fa = fb
            h *= 2.0
        if bracket_found:
            break

    if bracket_found:
        # We have a bracket [a, b] with fa * fb < 0
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Use bisection with a few secant attempts for speed
        # But primarily rely on bisection for guaranteed convergence
        for _ in range(100):
            if not time_ok():
                break

            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) < abs(best_f):
                best_x = b
                best_f = fb
            if abs(fa) < abs(best_f):
                best_x = a
                best_f = fa

            # Check if we've converged
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                # Return the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b

            # Try secant step
            if fb != fa:
                x_secant = b - fb * (b - a) / (fb - fa)
                # Use secant if it's inside the bracket
                if a < x_secant < b:
                    s = x_secant
                else:
                    s = 0.5 * (a + b)
            else:
                s = 0.5 * (a + b)

            fs = f_call(s)

            if not math.isfinite(fs):
                s = 0.5 * (a + b)
                fs = f_call(s)

            if abs(fs) <= 1e-8:
                return s
            if abs(fs) < abs(best_f):
                best_x = s
                best_f = fs

            # Update bracket
            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs

        return best_x
    else:
        # No bracket found. Use a derivative-free approach.
        # Since these are smooth polynomials, we can try to minimize |f(x)|
        # using a simple local search with adaptive step sizes.

        # Try a few Newton-like steps with finite differences
        x_curr = x
        fx_curr = fx
        h = 1e-6 * max(1.0, abs(x))

        for _ in range(50):
            if not time_ok():
                break

            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_curr) < abs(best_f):
                best_x = x_curr
                best_f = fx_curr

            # Finite difference derivative
            fx_plus = f_call(x_curr + h)
            if not math.isfinite(fx_plus):
                fx_plus = f_call(x_curr - h)
            if not math.isfinite(fx_plus):
                break

            d = (fx_plus - fx_curr) / h
            if d == 0 or not math.isfinite(d):
                break

            # Newton step
            x_new = x_curr - fx_curr / d

            if not math.isfinite(x_new):
                break

            # Damping: limit step size
            step = x_new - x_curr
            max_step = 5.0 * max(1.0, abs(x_curr))
            if abs(step) > max_step:
                x_new = x_curr + max_step * math.copysign(1.0, step)

            fx_new = f_call(x_new)

            if not math.isfinite(fx_new):
                break

            if abs(fx_new) < abs(best_f):
                best_x = x_new
                best_f = fx_new

            # Accept the step if it improves
            if abs(fx_new) < abs(fx_curr):
                x_curr = x_new
                fx_curr = fx_new
            else:
                # Reject: try a smaller step
                h *= 0.5
                if h < 1e-12:
                    break

        return best_x
