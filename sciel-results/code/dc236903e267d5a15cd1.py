"""Root finding: Ill-conditioned - combined approach with scipy brentq and 5-point Newton."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Combines 5-point Newton with backtracking, secant, and scipy brentq on brackets.
    """
    start_time = time.time()
    MAX_WALL = 140.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]

    def check_time():
        return (time.time() - start_time) < MAX_WALL

    def check_budget():
        return eval_count[0] < MAX_EVALS and check_time()

    def eval_f(x):
        if not check_budget():
            return 0.0
        eval_count[0] += 1
        try:
            val = f(x)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        return val

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def try_newton_fd(x0, max_iter=30):
        """Newton with 5-point finite difference and backtracking."""
        if not check_budget():
            return None
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x

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

            # Damping
            max_step = 0.5 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)

            x_new = x - dx
            if not math.isfinite(x_new):
                break

            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new

            # Backtracking if no improvement
            if abs(fx_new) >= abs(fx):
                improved = False
                for damp in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
                    if not check_budget():
                        break
                    x_d = x - damp * (x - x_new)
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                        improved = True
                        break
                if not improved:
                    break
            else:
                x, fx = x_new, fx_new

            if abs(dx) < 1e-12 * max(1.0, abs(x)):
                break

        return best_x[0] if best_f[0] <= TARGET_TOL else None

    def try_secant(x0, fx0, x1, fx1, max_iter=40):
        """Secant method with Illinois-style modification."""
        if not check_budget():
            return None
        x, fx = x1, fx1
        xp, fxp = x0, fx0

        for _ in range(max_iter):
            if not check_budget():
                break
            if is_good(x, fx):
                return x
            if fx == fxp:
                break
            d = (fx - fxp) / (x - xp)
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            # Illinois: if same side, shrink
            if fx * fx_new > 0:
                xp, fxp = x, fx
            else:
                xp, fxp = x, fx
                fxp *= 0.5
            x, fx = x_new, fx_new
            if abs(x_new - xp) < 1e-14 * max(1.0, abs(x_new)):
                break

        return best_x[0] if best_f[0] <= TARGET_TOL else None

    def find_bracket(x0, fx0, max_expansions=50):
        """Find a bracket [a, b] with sign change around x0."""
        if not check_budget():
            return None

        base_h = max(1e-4, 1e-4 * max(1.0, abs(x0)))

        # Symmetric expansion
        for i in range(max_expansions):
            if not check_budget():
                break
            h = base_h * (2 ** i)
            xl = x0 - h
            xr = x0 + h

            fxl = eval_f(xl)
            if is_good(xl, fxl):
                return (xl, xl, fxl, fxl)
            fxr = eval_f(xr)
            if is_good(xr, fxr):
                return (xr, xr, fxr, fxr)

            if fxl * fxr < 0:
                return (xl, xr, fxl, fxr)
            if fxl * fx0 < 0:
                return (xl, x0, fxl, fx0)
            if fx0 * fxr < 0:
                return (x0, xr, fx0, fxr)

            if h > 1e15:
                break

        # Asymmetric expansion
        for sign in [1, -1]:
            for i in range(max_expansions):
                if not check_budget():
                    break
                h = base_h * (2 ** i)
                x2 = x0 + sign * h
                fx2 = eval_f(x2)
                if is_good(x2, fx2):
                    return (x2, x2, fx2, fx2)
                if fx0 * fx2 < 0:
                    return (x0, x2, fx0, fx2)
                if h > 1e15:
                    break

        return None

    def try_scipy_brentq(a, b, fa, fb):
        """Use scipy.optimize.brentq for bracketed root finding."""
        if not check_budget():
            return None
        if a == b:
            return a if abs(fa) <= TARGET_TOL else None
        if fa * fb > 0:
            return None
        try:
            from scipy.optimize import brentq
            def f_wrap(x):
                if not check_budget():
                    raise RuntimeError("budget exhausted")
                return eval_f(x)
            root = brentq(f_wrap, a, b, xtol=1e-12, rtol=1e-12, maxiter=80)
            fr = eval_f(root)
            if is_good(root, fr):
                return root
            if abs(fr) < 1e-6:
                # Polish with secant
                h = 1e-4 * max(1.0, abs(root))
                x_prev = root - h
                fx_prev = eval_f(x_prev)
                result = try_secant(x_prev, fx_prev, root, fr, max_iter=10)
                if result is not None:
                    return result
            return None
        except Exception:
            return None

    def try_bisection(a, b, fa, fb, max_iter=60):
        """Bisection method as fallback."""
        if not check_budget():
            return None
        if fa * fb > 0:
            return None
        left, right = a, b
        fl, fr = fa, fb

        for _ in range(max_iter):
            if not check_budget():
                break
            mid = (left + right) / 2
            fm = eval_f(mid)
            if is_good(mid, fm):
                return mid
            if fl * fm <= 0:
                right, fr = mid, fm
            else:
                left, fl = mid, fm
            if abs(right - left) < 1e-14 * max(1.0, abs(mid)):
                break

        return best_x[0] if best_f[0] <= TARGET_TOL else None

    # Initial evaluation
    x = float(x0)
    fx = eval_f(x)
    if is_good(x, fx):
        return x

    # Strategy 1: 5-point Newton from x0
    result = try_newton_fd(x, max_iter=25)
    if result is not None:
        return result

    # Strategy 2: Find bracket and use scipy brentq
    bracket = find_bracket(x, fx, max_expansions=40)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a != b:
            result = try_scipy_brentq(a, b, fa, fb)
            if result is not None:
                return result
            result = try_bisection(a, b, fa, fb, max_iter=50)
            if result is not None:
                return result
        else:
            # Single point found to be good
            if abs(fa) <= TARGET_TOL:
                return a

    # Strategy 3: Secant with small perturbation
    h = 1e-5 * max(1.0, abs(x))
    fx_h = eval_f(x + h)
    result = try_secant(x, fx, x + h, fx_h, max_iter=30)
    if result is not None:
        return result

    h = 1e-5 * max(1.0, abs(x))
    fx_mh = eval_f(x - h)
    result = try_secant(x, fx, x - h, fx_mh, max_iter=30)
    if result is not None:
        return result

    # Strategy 4: Newton from perturbed points
    for factor in [0.5, 2.0, 0.1, 10.0]:
        x_try = x * factor if x != 0 else x + factor
        result = try_newton_fd(x_try, max_iter=20)
        if result is not None:
            return result

    # Strategy 5: Wider bracket search
    bracket = find_bracket(x, fx, max_expansions=50)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a != b:
            result = try_scipy_brentq(a, b, fa, fb)
            if result is not None:
                return result

    # Strategy 6: Secant with larger perturbation
    for h_factor in [1e-3, 1e-2, 1e-1]:
        h = h_factor * max(1.0, abs(x))
        fx_p = eval_f(x + h)
        result = try_secant(x, fx, x + h, fx_p, max_iter=25)
        if result is not None:
            return result
        fx_m = eval_f(x - h)
        result = try_secant(x, fx, x - h, fx_m, max_iter=25)
        if result is not None:
            return result

    # Return best found
    return best_x[0]

# EVOLVE-BLOCK-END
