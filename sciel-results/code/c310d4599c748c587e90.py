"""Root finding: Ill-conditioned - refined multi-strategy with tighter polish."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a SEARCH approach: try multiple varied strategies, keep the best result.
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

    def try_secant(x0, fx0, x1, fx1, max_iter=30):
        """Secant method."""
        if not check_budget():
            return None
        x, fx = x1, fx1
        xp, fxp = x0, fx0
        best_x_local = x
        best_f_local = abs(fx)

        for _ in range(max_iter):
            if not check_budget():
                break
            if fx == fxp:
                break
            d = (fx - fxp) / (x - xp)
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = eval_f(x_new)
            if abs(fx_new) < best_f_local:
                best_f_local = abs(fx_new)
                best_x_local = x_new
            if is_good(x_new, fx_new):
                return x_new
            xp, fxp = x, fx
            x, fx = x_new, fx_new
            if abs(x_new - xp) < 1e-14 * max(1.0, abs(x_new)):
                break

        return best_x_local if best_f_local <= TARGET_TOL else None

    def try_newton_fd(x0, max_iter=25):
        """Newton with 5-point finite difference, with damping."""
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
            if abs(fx_new) < best_f_local:
                best_f_local = abs(fx_new)
                best_x_local = x_new

            if is_good(x_new, fx_new):
                return x_new

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

    def try_brentq(a, b, fa, fb):
        """Use scipy.optimize.brentq."""
        if not check_budget():
            return None
        try:
            from scipy.optimize import brentq
            def f_wrap(x):
                return eval_f(x)
            root = brentq(f_wrap, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            fr = eval_f(root)
            if is_good(root, fr):
                return root
            if abs(fr) < 1e-6:
                h = 1e-4 * max(1.0, abs(root))
                x_prev = root - h
                fx_prev = eval_f(x_prev)
                return try_secant(x_prev, fx_prev, root, fr, max_iter=10)
            return None
        except Exception:
            return None

    def try_bisection(a, b, fa, fb, max_iter=80):
        """Bisection method."""
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
            if abs(fm) < best_f_local:
                best_f_local = abs(fm)
                best_x_local = mid
            if is_good(mid, fm):
                return mid
            if fl * fm <= 0:
                right, fr = mid, fm
            else:
                left, fl = mid, fm
            if abs(right - left) < 1e-14 * max(1.0, abs(mid)):
                break

        return best_x_local if best_f_local <= TARGET_TOL else None

    def try_illinois(a, b, fa, fb, max_iter=60):
        """Illinois method."""
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
            if abs(fl) < abs(fr):
                if fl != fr:
                    x_new = (fl * right - fr * left) / (fl - fr)
                    if not math.isfinite(x_new) or x_new <= left or x_new >= right:
                        x_new = (left + right) / 2
                else:
                    x_new = (left + right) / 2
            else:
                if fl != fr:
                    x_new = (fl * right - fr * left) / (fl - fr)
                    if not math.isfinite(x_new) or x_new <= left or x_new >= right:
                        x_new = (left + right) / 2
                else:
                    x_new = (left + right) / 2

            fx_new = eval_f(x_new)
            if abs(fx_new) < best_f_local:
                best_f_local = abs(fx_new)
                best_x_local = x_new
            if is_good(x_new, fx_new):
                return x_new

            if fl * fx_new <= 0:
                right = (right + x_new) / 2
                fr = eval_f(right)
                if abs(fr) < best_f_local:
                    best_f_local = abs(fr)
                    best_x_local = right
                if is_good(right, fr):
                    return right
            else:
                left = (left + x_new) / 2
                fl = eval_f(left)
                if abs(fl) < best_f_local:
                    best_f_local = abs(fl)
                    best_x_local = left
                if is_good(left, fl):
                    return left

            if abs(right - left) < 1e-14 * max(1.0, abs(x_new)):
                break

        return best_x_local if best_f_local <= TARGET_TOL else None

    def find_bracket(x0):
        """Adaptive bracket finding."""
        if not check_budget():
            return None

        fx0 = eval_f(x0)
        if is_good(x0, fx0):
            return (x0, x0, fx0, fx0)

        base_h = 1e-2 * max(1.0, abs(x0))

        scales = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]
        for scale in scales:
            if not check_budget():
                break
            h = base_h * scale
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

        for sign in [1, -1]:
            for scale in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 50000]:
                if not check_budget():
                    break
                h = base_h * scale
                x2 = x0 + sign * h
                fx2 = eval_f(x2)
                if is_good(x2, fx2):
                    return (x2, x2, fx2, fx2)
                if fx0 * fx2 < 0:
                    return (x0, x2, fx0, fx2)

        for scale in [1e6, 1e7, 1e8, 1e9]:
            if not check_budget():
                break
            h = max(1.0, abs(x0)) * scale
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

        return None

    def try_secant_from(x0):
        """Start secant from x0 with a small perturbation."""
        if not check_budget():
            return None
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        h = 1e-4 * max(1.0, abs(x))
        x_prev = x - h
        fx_prev = eval_f(x_prev)
        if is_good(x_prev, fx_prev):
            return x_prev
        return try_secant(x_prev, fx_prev, x, fx, max_iter=40)

    def try_minimize_scalar(x0):
        """Use scipy.optimize.minimize_scalar on f(x)^2 to find root."""
        if not check_budget():
            return None
        try:
            from scipy.optimize import minimize_scalar
            def obj(x):
                if not check_budget():
                    return 1e10
                fx = eval_f(x)
                return fx * fx

            base_h = max(1.0, abs(x0))
            for scale in [1, 10, 100, 1000, 10000]:
                if not check_budget():
                    break
                a = x0 - base_h * scale
                b = x0 + base_h * scale
                try:
                    res = minimize_scalar(obj, bounds=(a, b), method='bounded',
                                         options={'maxiter': 30, 'xatol': 1e-10})
                    if res.success and res.fun <= 1e-16:
                        x_sol = res.x
                        f_sol = eval_f(x_sol)
                        if is_good(x_sol, f_sol):
                            return x_sol
                        if abs(f_sol) < 1e-6:
                            h = 1e-4 * max(1.0, abs(x_sol))
                            x_prev = x_sol - h
                            fx_prev = eval_f(x_prev)
                            result = try_secant(x_prev, fx_prev, x_sol, f_sol, max_iter=10)
                            if result is not None:
                                return result
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def try_polish(x, fx):
        """High-precision polish using secant with very tight tolerance."""
        if not check_budget():
            return None
        if is_good(x, fx):
            return x
        # Use a small step for secant polish
        h = 1e-6 * max(1.0, abs(x))
        x_prev = x - h
        fx_prev = eval_f(x_prev)
        if is_good(x_prev, fx_prev):
            return x_prev
        result = try_secant(x_prev, fx_prev, x, fx, max_iter=20)
        if result is not None:
            return result
        # Try another direction
        x_prev2 = x + h
        fx_prev2 = eval_f(x_prev2)
        if is_good(x_prev2, fx_prev2):
            return x_prev2
        result = try_secant(x, fx, x_prev2, fx_prev2, max_iter=20)
        return result

    # === SEARCH STRATEGIES ===

    # Strategy 1: Bracket + Brent
    if check_budget():
        bracket = find_bracket(x0)
        if bracket is not None:
            a, b, fa, fb = bracket
            if a == b:
                return a
            if fa * fb < 0:
                result = try_brentq(a, b, fa, fb)
                if result is not None:
                    return result
                result = try_bisection(a, b, fa, fb)
                if result is not None:
                    return result

    # Strategy 2: Newton from x0
    if check_budget():
        result = try_newton_fd(x0, max_iter=25)
        if result is not None:
            return result

    # Strategy 3: Secant from x0
    if check_budget():
        result = try_secant_from(x0)
        if result is not None:
            return result

    # Strategy 4: Newton from perturbed points
    perturbations = [1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 5.0, -5.0]
    for offset in perturbations:
        if not check_budget():
            break
        xp = x0 + offset * max(1.0, abs(x0))
        result = try_newton_fd(xp, max_iter=15)
        if result is not None:
            return result

    # Strategy 5: Bracket from perturbed points
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
                result = try_brentq(a, b, fa, fb)
                if result is not None:
                    return result
                result = try_bisection(a, b, fa, fb)
                if result is not None:
                    return result

    # Strategy 6: Local scan for sign change
    if check_budget():
        base_h = 1e-3 * max(1.0, abs(x0))
        prev_x = x0 - base_h
        prev_fx = eval_f(prev_x)
        if is_good(prev_x, prev_fx):
            return prev_x
        for i in range(1, 100):
            if not check_budget():
                break
            cur_x = x0 + (i - 1) * base_h
            cur_fx = eval_f(cur_x)
            if is_good(cur_x, cur_fx):
                return cur_x
            if prev_fx * cur_fx < 0:
                result = try_brentq(prev_x, cur_x, prev_fx, cur_fx)
                if result is not None:
                    return result
                result = try_illinois(prev_x, cur_x, prev_fx, cur_fx)
                if result is not None:
                    return result
                result = try_bisection(prev_x, cur_x, prev_fx, cur_fx)
                if result is not None:
                    return result
            prev_x, prev_fx = cur_x, cur_fx

    # Strategy 7: Illinois method on any bracket found
    if check_budget():
        bracket = find_bracket(x0)
        if bracket is not None:
            a, b, fa, fb = bracket
            if a == b:
                return a
            if fa * fb < 0:
                result = try_illinois(a, b, fa, fb, max_iter=50)
                if result is not None:
                    return result

    # Strategy 7b: Ill-conditioned specific - try high precision polish
    if check_budget() and best_f[0] < 1e-4:
        x = best_x[0]
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        result = try_polish(x, fx)
        if result is not None:
            return result

    # Strategy 8: Try scipy minimize_scalar on f(x)^2
    if check_budget():
        result = try_minimize_scalar(x0)
        if result is not None:
            return result

    # Strategy 9: Try scipy fsolve or other methods as last resort
    if check_budget():
        try:
            from scipy.optimize import fsolve
            def f_wrap_scalar(x):
                return eval_f(x[0])
            sol, info, ier, msg = fsolve(f_wrap_scalar, [float(x0)], full_output=True, maxfev=50)
            if ier == 1 and len(sol) > 0:
                x_sol = float(sol[0])
                f_sol = eval_f(x_sol)
                if is_good(x_sol, f_sol):
                    return x_sol
                if abs(f_sol) < 1e-5:
                    result = try_polish(x_sol, f_sol)
                    if result is not None:
                        return result
        except Exception:
            pass

    # Strategy 10: Try multiple starting points with secant
    if check_budget():
        for offset in [1e-5, -1e-5, 1e-6, -1e-6]:
            if not check_budget():
                break
            xp = x0 + offset * max(1.0, abs(x0))
            result = try_secant_from(xp)
            if result is not None:
                return result

    # Final polish on best found
    if check_budget() and best_f[0] < 1e-3:
        x = best_x[0]
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        result = try_polish(x, fx)
        if result is not None:
            return result

    return best_x[0]

# EVOLVE-BLOCK-END
