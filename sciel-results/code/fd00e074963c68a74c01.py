import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses Illinois bisection for robustness, then a final Newton/secant polish.
    """
    start_time = time.time()
    deadline = start_time + 5.0
    tol = 1e-12
    max_iter = 150

    x0 = float(x0)
    f0 = f(x0)
    if abs(f0) <= tol:
        return x0

    # Find a bracket [a, b] with sign change
    base_step = 1e-4 * max(1.0, abs(x0))
    a = x0
    fa = f0
    b = None
    fb = None

    steps = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128]
    for s in steps:
        x_try = x0 + s * base_step
        f_try = f(x_try)
        if not math.isfinite(f_try):
            continue
        if abs(f_try) <= tol:
            return x_try
        if fa * f_try < 0:
            if abs(fa) < abs(f_try):
                a, b, fa, fb = x0, x_try, f0, f_try
            else:
                a, b, fa, fb = x_try, x0, f_try, f0
            break

    if b is not None:
        # Illinois bisection: converges faster than pure bisection on smooth functions
        prev_mid_side = None  # track which side we were on last iteration
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b
            if abs(b - a) < tol:
                return (a + b) / 2.0

            mid = (a + b) / 2.0
            f_mid = f(mid)

            if not math.isfinite(f_mid):
                # Fallback: try quartile points
                for frac in (0.25, 0.75, 0.1, 0.9):
                    mid = a + (b - a) * frac
                    f_mid = f(mid)
                    if math.isfinite(f_mid):
                        break
                else:
                    break

            if abs(f_mid) <= tol:
                return mid

            if fa * f_mid < 0:
                b, fb = mid, f_mid
                side = 'left'
            else:
                a, fa = mid, f_mid
                side = 'right'

            # Illinois: if we stayed on the same side, shrink the other endpoint
            if prev_mid_side == side:
                if side == 'left':
                    # Root was on left last time too; shrink right endpoint
                    b = b - (b - a) * 0.5
                    fb = f(b)
                    if abs(fb) <= tol:
                        return b
                else:
                    # Root was on right last time too; shrink left endpoint
                    a = a + (b - a) * 0.5
                    fa = f(a)
                    if abs(fa) <= tol:
                        return a
            prev_mid_side = side

        # Pick the better endpoint
        best_x = a if abs(fa) < abs(fb) else b
        best_f = min(abs(fa), abs(fb))
    else:
        # No bracket: use secant method
        x_prev = x0
        f_prev = f0
        x_curr = x0 + base_step
        f_curr = f(x_curr)
        if abs(f_curr) <= tol:
            return x_curr

        best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
        best_f = min(abs(f_prev), abs(f_curr))

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                break
            x_next = x_curr - f_curr * (x_curr - x_prev) / denom
            f_next = f(x_next)
            if not math.isfinite(f_next) or not math.isfinite(x_next):
                break
            if abs(f_next) < best_f:
                best_x, best_f = x_next, abs(f_next)
            if abs(f_next) <= tol:
                return x_next
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next

    # Final polish: try a few Newton/secant steps from best_x to tighten
    # This can shave off a few bisection calls worth of precision
    x_best = best_x
    f_best = f(x_best) if best_x != x0 else f0
    # Use secant from two nearby points for the polish
    dx = 1e-6 * max(1.0, abs(x_best))
    x2 = x_best + dx
    f2 = f(x2)
    if abs(f2) < abs(f_best):
        x_best, f_best = x2, f2

    for _ in range(10):
        if time.time() > deadline - 0.5:
            break
        if abs(f_best) <= tol:
            return x_best
        # Secant step using (x_best, f_best) and (x2, f2)
        denom = f_best - f2
        if abs(denom) < 1e-15:
            break
        x_new = x_best - f_best * (x_best - x2) / denom
        f_new = f(x_new)
        if not math.isfinite(f_new) or not math.isfinite(x_new):
            break
        if abs(f_new) < abs(f_best):
            x2, f2 = x_best, f_best
            x_best, f_best = x_new, f_new
        else:
            # If secant diverges, just break
            break

    return x_best
