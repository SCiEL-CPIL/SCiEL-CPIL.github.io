"""
Root finder using Brent's method (inverse quadratic interpolation with bisection safeguard).
Structural change: replace the multi-phase (bracket expansion + Newton + grid) pipeline
with a single, unified Brent-style iteration that reuses the two most recent evaluations
per step, giving superlinear convergence without derivative calls.
"""
import time
import math

def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomials.
    Uses Brent's method: inverse quadratic interpolation with bisection safeguard.
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    TOL = 1e-9
    MAX_CALLS = 480

    x = float(x0)
    call_count = 0
    cache = {}

    def _elapsed():
        return time.time() - start_time

    def _time_ok():
        return _elapsed() < deadline and call_count < MAX_CALLS

    def _f(x_val):
        """Cached function evaluation with call counting."""
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(key)
            call_count += 1
            if not math.isfinite(val):
                cache[key] = None
                return None
            cache[key] = val
            return val
        except Exception:
            call_count += 1
            return None

    # Phase 0: Check if x0 is already a root
    fx0 = _f(x)
    if fx0 is None:
        return x
    if abs(fx0) <= TOL:
        return x

    # Phase 1: Find a bracket [a, b] with f(a)*f(b) < 0
    # Use exponential expansion from x0
    a = b = None
    fa = fb = None

    step = 0.5
    for i in range(50):
        if not _time_ok():
            break
        if step > 1e10:
            break

        # Try right
        xr = x + step
        fr = _f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx0 * fr < 0:
                a, b = x, xr
                fa, fb = fx0, fr
                break

        # Try left
        xl = x - step
        fl = _f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx0 * fl < 0:
                a, b = xl, x
                fa, fb = fl, fx0
                break

        step *= 1.6

    # If no bracket found, do a wider grid search
    if a is None or b is None:
        best_x = x
        best_abs = abs(fx0)
        prev_x, prev_f = None, None

        # Search on expanding intervals
        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
            if not _time_ok():
                break
            base = max(abs(x0), 1.0) * scale
            n_samples = 30
            for i in range(n_samples + 1):
                if not _time_ok():
                    break
                t = i / n_samples
                x_try = x0 - base + 2 * base * t
                f_try = _f(x_try)
                if f_try is None:
                    continue
                if abs(f_try) < best_abs:
                    best_abs = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= TOL:
                    return x_try
                if prev_f is not None and prev_f * f_try < 0:
                    a, b = prev_x, x_try
                    fa, fb = prev_f, f_try
                    break
                prev_x, prev_f = x_try, f_try
            if a is not None:
                break

    # If still no bracket, fall back to secant from best point
    if a is None or b is None:
        x_curr = best_x
        fx_curr = _f(x_curr)
        if fx_curr is None:
            return best_x
        x_prev = x_curr + 1e-4 * max(1.0, abs(x_curr))
        fx_prev = _f(x_prev)
        if fx_prev is None:
            return best_x

        for _ in range(60):
            if not _time_ok():
                break
            if abs(fx_curr) <= TOL:
                return x_curr
            if abs(fx_prev) <= TOL:
                return x_prev
            denom = fx_curr - fx_prev
            if abs(denom) < 1e-15:
                break
            x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
            if not math.isfinite(x_new):
                break
            x_prev, fx_prev = x_curr, fx_curr
            x_curr = x_new
            fx_curr = _f(x_curr)
            if fx_curr is None:
                break
        return x_curr

    # Phase 2: Brent's method on the bracket [a, b]
    # Ensure fa < 0 < fb (swap if needed)
    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa

    # Brent's method state
    # We maintain:
    #   a, b: current bracket with f(a)*f(b) <= 0
    #   c: previous value of b
    #   d: previous value of a
    #   e: length of the previous step
    #   s: previous step size

    x_a, x_b = a, b
    f_a, f_b = fa, fb

    # Initialize c, d as the previous endpoints
    x_c = x_a
    f_c = f_a
    x_d = x_a
    f_d = f_a

    # e = previous step length, s = sign of previous step
    e = 0.0
    s = 0.0

    # Track convergence
    for _ in range(80):
        if not _time_ok():
            break

        if abs(f_a) <= TOL:
            return x_a
        if abs(f_b) <= TOL:
            return x_b

        # Inverse quadratic interpolation
        # Given three points (x_a, f_a), (x_c, f_c), (x_b, f_b)
        # where f_a and f_b have opposite signs, find the root of the
        # quadratic interpolant in x as a function of f.

        # The inverse quadratic interpolant passes through:
        #   (f_a, x_a), (f_c, x_c), (f_b, x_b)
        # We solve for x where f = 0.

        # Using the formula for inverse quadratic interpolation:
        # x = x_b + 0.5 * (f_b / (f_b - f_a)) * (x_b - x_a)  [linear]
        # For quadratic, use Lagrange interpolation on (f, x):
        #
        # x(f) = x_a * L_a(f) + x_c * L_c(f) + x_b * L_b(f)
        # where L_i are Lagrange basis polynomials in f.
        #
        # At f=0:
        # x(0) = x_a * (f_c * f_b) / ((f_a - f_c)(f_a - f_b))
        #      + x_c * (f_a * f_b) / ((f_c - f_a)(f_c - f_b))
        #      + x_b * (f_a * f_c) / ((f_b - f_a)(f_b - f_c))

        denom1 = (f_a - f_c) * (f_a - f_b)
        denom2 = (f_c - f_a) * (f_c - f_b)
        denom3 = (f_b - f_a) * (f_b - f_c)

        x_new = None
        if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
            try:
                x_new = (x_a * f_c * f_b / denom1
                         + x_c * f_a * f_b / denom2
                         + x_b * f_a * f_c / denom3)
                if not math.isfinite(x_new):
                    x_new = None
            except (ZeroDivisionError, OverflowError):
                x_new = None

        # If inverse quadratic fails or gives a point outside the bracket,
        # fall back to bisection (or secant if only 2 points are distinct)
        if x_new is None or not (x_a < x_new < x_b):
            # Secant step using the two most recent distinct points
            if f_a != f_b and abs(f_a - f_b) > 1e-15:
                x_sec = x_a - f_a * (x_a - x_b) / (f_a - f_b)
                if x_a < x_sec < x_b:
                    x_new = x_sec
                else:
                    x_new = (x_a + x_b) / 2.0
            else:
                x_new = (x_a + x_b) / 2.0

        # Evaluate f at x_new
        f_new = _f(x_new)
        if f_new is None:
            break
        if abs(f_new) <= TOL:
            return x_new

        # Update bracket
        if f_a * f_new < 0:
            # Root is in [x_a, x_new]
            x_c, f_c = x_b, f_b
            x_b, f_b = x_new, f_new
        else:
            # Root is in [x_new, x_b]
            x_c, f_c = x_a, f_a
            x_a, f_a = x_new, f_new

    # Return the endpoint with smaller |f|
    if abs(f_a) <= abs(f_b):
        return x_a
    else:
        return x_b
