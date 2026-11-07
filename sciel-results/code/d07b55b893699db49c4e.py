"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Strategy:
      1. Try to find a bracket around x0 via adaptive expansion.
      2. Use Brent's method (bisection + secant + inverse quadratic interpolation)
         when a bracket is found.
      3. If no bracket found, fall back to a damped Newton / secant hybrid.
      4. Time guard: return best candidate before 110s wall-clock.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    x0 = float(x0)
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # --- Phase 1: Try to find a bracket around x0 ---
    # Expand radius adaptively to find sign change
    a, b = None, None
    fa, fb = None, None
    best_x = x0
    best_fabs = abs(fx0)

    radius = 1.0
    max_radius = 1e10
    for _ in range(60):
        if elapsed() > deadline - 5.0:
            break
        # Try x0 - radius and x0 + radius
        for sign in (-1, 1):
            xt = x0 + sign * radius
            try:
                ft = f(xt)
            except Exception:
                continue
            if not math.isfinite(ft):
                continue
            if abs(ft) < best_fabs:
                best_fabs = abs(ft)
                best_x = xt
            if abs(ft) <= 1e-8:
                return xt
            # Check bracket
            if a is None or (fa is not None and fa * ft < 0):
                # We have a sign change between x0 and xt if fx0 * ft < 0
                pass
            if fx0 * ft < 0:
                if sign == -1:
                    a, b = xt, x0
                    fa, fb = ft, fx0
                else:
                    a, b = x0, xt
                    fa, fb = fx0, ft
                break
        if a is not None:
            break
        radius *= 2.0
        if radius > max_radius:
            break

    # --- Phase 2: Brent's method if bracket found ---
    if a is not None and b is not None:
        return _brent(f, a, b, fa, fb, deadline, elapsed)

    # --- Phase 3: No bracket found. Use secant/Newton hybrid with damping ---
    # Start from best_x if it's better
    x = best_x
    fx = f(x) if x != x0 else fx0
    if abs(fx) <= 1e-8:
        return x

    # Secant method with two points
    # Use x0 and x0 + small perturbation
    h = 1e-4 * max(1.0, abs(x0))
    x_prev = x0
    fx_prev = fx0
    x_curr = x
    fx_curr = fx

    for _ in range(200):
        if elapsed() > deadline - 2.0:
            break
        if abs(fx_curr) < abs(fx_prev):
            x_prev, fx_prev = x_curr, fx_curr
        if abs(fx_curr) <= 1e-8:
            return x_curr

        # Secant step
        if fx_curr != fx_prev:
            x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        else:
            # Newton-like step with finite difference
            h2 = 1e-6 * max(1.0, abs(x_curr))
            try:
                d = (f(x_curr + h2) - fx_curr) / h2
            except Exception:
                break
            if d == 0 or not math.isfinite(d):
                break
            x_next = x_curr - fx_curr / d

        if not math.isfinite(x_next):
            break

        # Damping: if step is too large, reduce it
        step = x_next - x_curr
        if abs(step) > 10.0 * max(1.0, abs(x_curr)):
            x_next = x_curr + math.copysign(min(abs(step), 10.0 * max(1.0, abs(x_curr))), step)

        try:
            fx_next = f(x_next)
        except Exception:
            break

        if not math.isfinite(fx_next):
            break

        if abs(fx_next) < best_fabs:
            best_fabs = abs(fx_next)
            best_x = x_next

        if abs(fx_next) <= 1e-8:
            return x_next

        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_next, fx_next

    # Return best found
    return best_x


def _brent(f, a, b, fa, fb, deadline, elapsed):
    """
    Brent's method for finding a root in [a, b] where f(a)*f(b) < 0.
    Uses bisection, secant, and inverse quadratic interpolation.
    """
    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d
    max_iter = 100

    for _ in range(max_iter):
        if elapsed() > deadline - 2.0:
            break
        if abs(fa) <= 1e-8:
            return a
        if fb * fa > 0:
            # Shouldn't happen, but reset
            a, c = b, a
            fa, fc = fb, fa
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            d = b - a
            e = d
            continue

        # Check for convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(b)):
            return b if abs(fb) < abs(fa) else a

        # Inverse quadratic interpolation
        try:
            # Compute coefficients for inverse quadratic interpolation
            # Using the standard Brent formulation
            if e != 0 and fc != 0:
                # Inverse quadratic interpolation
                t1 = (b / (b - c)) * (fb / (fb - fc))
                t2 = (b / (b - a)) * (fb / (fb - fa))
                s = t1 * t2
                # Check if IQU is acceptable
                if not (0.5 < s < 1.5):
                    # Use secant instead
                    s = (b - a) * (fb / (fb - fa))
                else:
                    s = (b - a) * (fb / (fb - fa)) * (t1 - 1) * (t2 - 1) / (t1 * t2)
            else:
                # Secant
                s = (b - a) * (fb / (fb - fa))
        except Exception:
            s = (b - a) * (fb / (fb - fa))

        # Check if secant/iqu step is acceptable
        if not (0.5 * abs(e) < abs(s) < abs(e)):
            # Bisection
            s = 0.5 * (b - a)

        # Move
        e = d
        d = s

        if abs(s) < 1e-14:
            # Very small step, use bisection direction
            s = 0.5 * (b - a) if s >= 0 else -0.5 * (b - a)
            if s == 0:
                s = 1e-14

        try:
            fa_new = f(a + s)
        except Exception:
            break

        if not math.isfinite(fa_new):
            break

        if abs(fa_new) <= 1e-8:
            return a + s

        # Update bracket
        c, fc = b, fb
        if fa * fa_new < 0:
            b, fb = a, fa
        else:
            a = a + s
            fa = fa_new

        # Ensure |fa| <= |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    # Return the endpoint with smaller |f|
    return a if abs(fa) <= abs(fb) else b

# EVOLVE-BLOCK-END
