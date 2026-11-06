"""Root finding: Ill-conditioned - Illinois method with adaptive bracketing and polish."""
import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses Illinois method (modified secant with 0.5 shrinkage) as primary solver,
    with scipy brentq as fallback, plus final high-precision polish.
    """
    start_time = time.time()
    MAX_WALL = 145.0  # Safety margin before 150s kill
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        eval_count[0] += 1
        val = f(x)
        if not math.isfinite(val):
            val = 0.0
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        return val

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def check_time():
        return time.time() - start_time < MAX_WALL

    def check_budget():
        return eval_count[0] < MAX_EVALS and check_time()

    # Illinois method: modified secant that shrinks larger sub-interval by factor 0.5
    def try_illinois(a, b, fa, fb, max_iter=200):
        """
        Illinois method for ill-conditioned functions.
        Uses secant update but when the new point doesn't change sign,
        it shrinks the LARGER sub-interval by a factor of 0.5 instead of
        replacing it with the secant point (which is what standard bisection does).
        This gives faster convergence than bisection while maintaining robustness.
        """
        if not check_budget():
            return None
        if fa * fb > 0:
            return None

        left, right = a, b
        fl, fr = fa, fb
        best_x_local = left if abs(fl) < abs(fr) else right
        best_f_local = min(abs(fl), abs(fr))

        for _ in range(max_iter):
            if not check_budget():
                break

            # Secant point
            if fl != fr:
                x_sec = (left * fr - right * fl) / (fr - fl)
            else:
                x_sec = (left + right) / 2

            # Clamp secant point to bracket
            if x_sec < left or x_sec > right:
                x_sec = (left + right) / 2

            fx = eval_f(x_sec)
            if is_good(x_sec, fx):
                return x_sec
            if abs(fx) < best_f_local:
                best_f_local = abs(fx)
                best_x_local = x_sec

            # Check for sign change with left
            if fl * fx < 0:
                # Root is in [left, x_sec]
                right = x_sec
                fr = fx
            elif fl * fx > 0:
                # Root is in [x_sec, right]
                left = x_sec
                fl = fx
            else:
                # fx is zero (or very close)
                return x_sec

            # Illinois modification: shrink the LARGER sub-interval by 0.5
            # This is the key difference from standard bisection
            if abs(right - x_sec) > abs(x_sec - left):
                # The right sub-interval is larger; shrink it
                new_left = x_sec + 0.5 * (right - x_sec)
                if new_left < right:
                    left = new_left
                    fl = eval_f(left)
                    if is_good(left, fl):
                        return left
                    if abs(fl) < best_f_local:
                        best_f_local = abs(fl)
                        best_x_local = left
            else:
                # The left sub-interval is larger; shrink it
                new_right = x_sec - 0.5 * (x_sec - left)
                if new_right > left:
                    right = new_right
                    fr = eval_f(right)
                    if is_good(right, fr):
                        return right
                    if abs(fr) < best_f_local:
                        best_f_local = abs(fr)
                        best_x_local = right

            # Convergence check
            if abs(right - left) < 1e-14 * max(1.0, abs((left + right) / 2)):
                break

        # Return best point found
        if best_f_local <= TARGET_TOL:
            return best_x_local
        # Check endpoints one more time
        if abs(fl) <= TARGET_TOL:
            return left
        if abs(fr) <= TARGET_TOL:
            return right
        return None

    # Try to find a bracket [a, b] where f(a)*f(b) < 0
    def find_bracket(x0):
        """Adaptive bracket finding with step-doubling around x0."""
        if not check_budget():
            return None

        fx0 = eval_f(x0)
        if is_good(x0, fx0):
            return (x0, x0, fx0, fx0)

        # Step-doubling: start small, double until sign change found
        base_h = 1e-3 * max(1.0, abs(x0))
        h = base_h

        for _ in range(60):  # Up to 2^60 * base_h range
            if not check_budget():
                break
            xl = x0 - h
            xr = x0 + h

            fxl = eval_f(xl)
            if is_good(xl, fxl):
                return (xl, xl, fxl, fxl)

            fxr = eval_f(xr)
            if is_good(xr, fxr):
                return (xr, xr, fxr, fxr)

            # Check for sign changes
            if fxl * fx0 < 0:
                return (xl, x0, fxl, fx0)
            if fx0 * fxr < 0:
                return (x0, xr, fx0, fxr)
            if fxl * fxr < 0:
                return (xl, xr, fxl, fxr)

            h *= 2.0

        # Try asymmetric expansion
        for sign in [1, -1]:
            h = base_h
            for _ in range(50):
                if not check_budget():
                    break
                x2 = x0 + sign * h
                fx2 = eval_f(x2)
                if is_good(x2, fx2):
                    return (x2, x2, fx2, fx2)
                if fx0 * fx2 < 0:
                    return (x0, x2, fx0, fx2)
                h *= 2.0

        return None

    # Brent's method via scipy (fallback)
    def try_brentq(a, b, fa, fb):
        """Use scipy.optimize.brentq with our counted eval."""
        if not check_budget():
            return None
        try:
            def f_wrap(x):
                return eval_f(x)
            root = brentq(f_wrap, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            fr = eval_f(root)
            if is_good(root, fr):
                return root
            if abs(fr) < 1e-6:
                return try_polish(root, fr)
            return None
        except Exception:
            return None

    # Secant polish for final precision
    def try_polish(x0, fx0, max_iter=15):
        """Secant method polish starting from a near-root."""
        if not check_budget():
            return None
        x = float(x0)
        fx = fx0
        if is_good(x, fx):
            return x

        h = 1e-5 * max(1.0, abs(x))
        x_prev = x - h
        fx_prev = eval_f(x_prev)
        if is_good(x_prev, fx_prev):
            return x_prev

        for _ in range(max_iter):
            if not check_budget():
                break
            if fx == fx_prev:
                break
            d = (fx - fx_prev) / (x - x_prev)
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
            if abs(x_new - x_prev) < 1e-15 * max(1.0, abs(x_new)):
                break

        return x if abs(fx) <= TARGET_TOL else None

    # Newton with 5-point derivative and damping
    def try_newton(x0, max_iter=15):
        """Newton's method with 5-point FD derivative and damping."""
        if not check_budget():
            return None
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x

        best_x_local = x
        best_f_local = abs(fx)

        for _ in range(max_iter):
            if not check_budget():
                break

            h = (1e-7) ** 0.25 * max(1.0, abs(x))
            h = max(h, 1e-8)

            fx_m2 = eval_f(x - 2*h)
            if is_good(x - 2*h, fx_m2):
                return x - 2*h
            fx_m1 = eval_f(x - h)
            if is_good(x - h, fx_m1):
                return x - h
            fx_p1 = eval_f(x + h)
            if is_good(x + h, fx_p1):
                return x + h
            fx_p2 = eval_f(x + 2*h)
            if is_good(x + 2*h, fx_p2):
                return x + 2*h

            d = (-fx_p2 + 8*fx_p1 - 8*fx_m1 + fx_m2) / (12*h)
            if d == 0 or not math.isfinite(d):
                break

            dx = fx / d
            if not math.isfinite(dx):
                break

            max_step = 0.5 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)

            x_new = x - dx
            if not math.isfinite(x_new):
                break

            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new

            if abs(fx_new) < best_f_local:
                best_f_local = abs(fx_new)
                best_x_local = x_new

            if abs(fx_new) >= abs(fx):
                improved = False
                for damp in [0.5, 0.25, 0.125, 0.0625]:
                    if not check_budget():
                        break
                    x_d = x - damp * (x - x_new)
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                        if abs(fx) < best_f_local:
                            best_f_local = abs(fx)
                            best_x_local = x
                        improved = True
                        break
                if not improved:
                    break
            else:
                x, fx = x_new, fx_new

            if abs(dx) < 1e-12 * max(1.0, abs(x)):
                break

        return best_x_local if best_f_local <= TARGET_TOL else None

    # Bisection fallback
    def try_bisection(a, b, fa, fb, max_iter=80):
        if not check_budget():
            return None
        if fa * fb > 0:
            return None
        left, right = a, b
        fl, fr = fa, fb
        best_x_local = left if abs(fl) < abs(fr) else right
        best_f_local = min(abs(fl), abs(fr))

        for _ in range(max_iter):
            if not check_budget():
                break
            mid = (left + right) / 2
            fm = eval_f(mid)
            if is_good(mid, fm):
                return mid
            if abs(fm) < best_f_local:
                best_f_local = abs(fm)
                best_x_local = mid
            if fl * fm <= 0:
                right, fr = mid, fm
            else:
                left, fl = mid, fm
            if abs(right - left) < 1e-14 * max(1.0, abs(mid)):
                break

        return best_x_local if best_f_local <= TARGET_TOL else None

    # === MAIN STRATEGY ===

    # Step 1: Find bracket and try Illinois method (primary for ill-conditioned)
    bracket = find_bracket(x0)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a == b:
            return a
        if fa * fb < 0:
            # Try Illinois first (designed for ill-conditioned)
            result = try_illinois(a, b, fa, fb)
            if result is not None:
                return result
            # Fallback to brentq
            result = try_brentq(a, b, fa, fb)
            if result is not None:
                return result
            # Fallback to bisection
            result = try_bisection(a, b, fa, fb)
            if result is not None:
                return result

    # Step 2: Try Newton from x0
    result = try_newton(x0)
    if result is not None:
        return result

    # Step 3: Try secant from x0
    if check_budget():
        x = float(x0)
        fx = eval_f(x)
        if not is_good(x, fx):
            h = 1e-4 * max(1.0, abs(x))
            x_prev = x - h
            fx_prev = eval_f(x_prev)
            if not is_good(x_prev, fx_prev):
                for _ in range(40):
                    if not check_budget():
                        break
                    if fx == fx_prev:
                        break
                    d = (fx - fx_prev) / (x - x_prev)
                    if d == 0 or not math.isfinite(d):
                        break
                    x_new = x - fx / d
                    if not math.isfinite(x_new):
                        break
                    fx_new = eval_f(x_new)
                    if is_good(x_new, fx_new):
                        return x_new
                    x_prev, fx_prev = x, fx
                    x, fx = x_new, fx_new
                    if abs(fx) < best_f[0]:
                        best_f[0] = abs(fx)
                        best_x[0] = x

    # Step 4: Try Newton from perturbed points
    for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
        if not check_budget():
            break
        xp = x0 + offset * max(1.0, abs(x0))
        result = try_newton(xp, max_iter=15)
        if result is not None:
            return result

    # Step 5: Try to find bracket from perturbed points, use Illinois
    for offset in [1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
        if not check_budget():
            break
        xp = x0 + offset * max(1.0, abs(x0))
        bracket = find_bracket(xp)
        if bracket is not None:
            a, b, fa, fb = bracket
            if a == b:
                return a
            if fa * fb < 0:
                result = try_illinois(a, b, fa, fb)
                if result is not None:
                    return result
                result = try_brentq(a, b, fa, fb)
                if result is not None:
                    return result
                result = try_bisection(a, b, fa, fb)
                if result is not None:
                    return result

    # Step 6: Local scan for sign change
    if check_budget():
        base_h = 1e-3 * max(1.0, abs(x0))
        prev_x = x0 - base_h
        prev_fx = eval_f(prev_x)
        for i in range(1, 100):
            if not check_budget():
                break
            cur_x = x0 + (i - 1) * base_h
            cur_fx = eval_f(cur_x)
            if is_good(cur_x, cur_fx):
                return cur_x
            if prev_fx * cur_fx < 0:
                result = try_illinois(prev_x, cur_x, prev_fx, cur_fx)
                if result is not None:
                    return result
                result = try_brentq(prev_x, cur_x, prev_fx, cur_fx)
                if result is not None:
                    return result
                result = try_bisection(prev_x, cur_x, prev_fx, cur_fx)
                if result is not None:
                    return result
            prev_x, prev_fx = cur_x, cur_fx

    # Step 7: Final polish - if we have a good candidate, try to improve it
    if check_budget() and best_f[0] < 1e-4:
        result = try_polish(best_x[0], f(best_x[0]), max_iter=20)
        if result is not None:
            return result

    # Return best found
    return best_x[0]

# EVOLVE-BLOCK-END
