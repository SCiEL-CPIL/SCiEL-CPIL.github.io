"""Combined root finder: Newton with forward-diff (call-efficient) + bracketing fallback + polish."""
import math
import time

_DEADLINE = time.monotonic() + 140.0  # safety margin before 150s kill


def _check_deadline():
    return time.monotonic() < _DEADLINE


def find_root(f, x0):
    x = float(x0)
    best_x = x
    best_fx = None

    # Evaluate initial
    try:
        fx = f(x)
    except Exception:
        return x
    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    best_fx = fx
    best_x = x

    # Phase 1: Secant Method (1 call per iteration after initial setup)
    # We need two distinct points to start the Secant method.
    # We have x0, f(x0). Let's find x1 close to x0.
    x1 = x + 1e-4 * max(1.0, abs(x))
    try:
        fx1 = f(x1)
    except Exception:
        # If we can't even get a second point, fall back to bracketing immediately
        if _check_deadline():
            x_bracketed = _bracket_and_solve(f, x, fx)
            if x_bracketed is not None:
                try:
                    fx_br = f(x_bracketed)
                    if math.isfinite(fx_br) and abs(fx_br) < abs(fx):
                        return x_bracketed
                except Exception:
                    pass
        return x

    if not math.isfinite(fx1):
        if _check_deadline():
            x_bracketed = _bracket_and_solve(f, x, fx)
            if x_bracketed is not None:
                try:
                    fx_br = f(x_bracketed)
                    if math.isfinite(fx_br) and abs(fx_br) < abs(fx):
                        return x_bracketed
                except Exception:
                    pass
        return x

    # Update best if fx1 is better
    if abs(fx1) < abs(fx):
        best_x = x1
        best_fx = fx1

    # Now we have (x, fx) and (x1, fx1). 
    # Secant iteration:
    # x_new = x - fx * (x - x1) / (fx - fx1)
    
    a = x
    fa = fx
    b = x1
    fb = fx1

    converged = False
    for i in range(50):
        if not _check_deadline():
            break
        
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b

        denom = fb - fa
        if abs(denom) < 1e-300:
            # Points are too close or function is flat, try a larger step
            step = 1e-4 * max(1.0, abs(b))
            c = b + step
            try:
                fc = f(c)
            except Exception:
                break
            if not math.isfinite(fc):
                break
            a = b
            fa = fb
            b = c
            fb = fc
            continue

        c = b - fb * (b - a) / (fb - fa)

        # Damping: if step is huge, limit it to prevent divergence
        if abs(c - b) > 10.0 * max(1.0, abs(b)):
            c = b + 10.0 * max(1.0, abs(b)) * (1 if c > b else -1)

        if not math.isfinite(c):
            break

        try:
            fc = f(c)
        except Exception:
            break
        if not math.isfinite(fc):
            break

        if abs(fc) < abs(best_fx):
            best_x = c
            best_fx = fc

        # Shift points: (a, b) -> (b, c)
        a = b
        fa = fb
        b = c
        fb = fc

        if abs(fc) <= 1e-8:
            converged = True
            break

        # If convergence is slow or oscillating, check if we should break to bracketing
        # But Secant is generally robust for polynomials.
        if i > 20 and abs(b - a) < 1e-12 * max(1.0, abs(b)):
             break

    if converged:
        return b if abs(fb) <= 1e-8 else a

    # Phase 2: If Secant didn't converge, try bracketing + bisection/interpolation
    # We should start bracketing from the best point found so far
    if abs(best_fx) > 1e-8 and _check_deadline():
        x_bracketed = _bracket_and_solve(f, best_x, best_fx)
        if x_bracketed is not None:
            try:
                fx_br = f(x_bracketed)
                if math.isfinite(fx_br) and abs(fx_br) < abs(best_fx):
                    best_x = x_bracketed
                    best_fx = fx_br
            except Exception:
                pass

    # Phase 3: High-precision polish if close
    if best_fx is not None and abs(best_fx) <= 1e-6 and _check_deadline():
        x = _polish(f, best_x, best_fx)
        try:
            fx_check = f(x)
            if math.isfinite(fx_check) and abs(fx_check) < abs(best_fx):
                best_x = x
                best_fx = fx_check
        except Exception:
            pass

    return best_x


def _polish(f, x, fx):
    """High-precision Newton polish with central difference and tighter tolerance."""
    for _ in range(20):
        if not _check_deadline():
            break
        if not math.isfinite(x) or fx is None or not math.isfinite(fx):
            break
        if abs(fx) <= 1e-12:
            break

        h = 1e-8 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except Exception:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break

        d = (f_xp - f_xm) / (2.0 * h)
        if abs(d) < 1e-16:
            break

        x_new = x - fx / d
        if not math.isfinite(x_new):
            break

        # Small damping for polish
        step = x_new - x
        if abs(step) > 0.1:
            x_new = x + (0.1 if step > 0 else -0.1)

        x = x_new
        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break

    return x


def _bracket_and_solve(f, x_start, fx_start):
    """Try to find a bracket and solve using bisection/interpolation."""
    if not _check_deadline():
        return None
    if fx_start is None or not math.isfinite(fx_start):
        return None

    a = x_start
    fa = fx_start
    step = 1.0
    b = None
    fb = None

    # Expand to find a sign change
    for _ in range(30):
        if not _check_deadline():
            break
        b = a + step
        try:
            fb = f(b)
        except Exception:
            step *= 2.0
            continue
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        # If no sign change, try the other direction
        b = a - step
        try:
            fb = f(b)
        except Exception:
            step *= 2.0
            continue
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        step *= 2.0
    else:
        return None

    # Now we have a bracket [a, b] with fa * fb <= 0
    # Ensure a is the left endpoint
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))

    # Safeguarded inverse interpolation
    for _ in range(50):
        if not _check_deadline():
            break
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a

        denom = fb - fa
        if abs(denom) < 1e-300:
            x_new = (a + b) / 2.0
        else:
            x_new = (a * fb - b * fa) / denom

        # Safeguard: x_new must be in (a, b)
        if not (a < x_new < b):
            x_new = (a + b) / 2.0

        try:
            f_new = f(x_new)
        except Exception:
            x_new = (a + b) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
        if not math.isfinite(f_new):
            x_new = (a + b) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        if abs(f_new) < best_f:
            best_x = x_new
            best_f = abs(f_new)

        if abs(f_new) <= 1e-8:
            return x_new

        # Update bracket
        if fa * f_new <= 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new

        # Check convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

    if abs(fb) <= abs(fa):
        return b
    else:
        return a
