"""Ill-conditioned root finding with adaptive multi-strategy approach."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 128.0
    max_evals = 500
    TOL = 1e-8
    eval_count = 0
    best_x = float(x0)
    best_fabs = abs(f(best_x))
    eval_count += 1

    def call_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            raise Exception("max evals")
        if time.time() > deadline:
            raise Exception("deadline")
        val = f(x)
        eval_count += 1
        if not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = float('nan')
        fabs = abs(val)
        if math.isfinite(fabs) and fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        return val

    def check_time():
        return time.time() < deadline

    def secant_with_restart(x0, tol=TOL, max_iter=150):
        """Secant method with adaptive restarts on stall."""
        x1 = float(x0)
        h = 1e-5 * max(1.0, abs(x1))
        x2 = x1 + h

        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False

        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True

        prev_fabs = abs(f2)
        stall_count = 0

        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True

            denom = f2 - f1
            if abs(denom) < 1e-30:
                h2 = 1e-5 * max(1.0, abs(x2)) * (1 + 0.1 * (i % 10))
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False

            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                # Restart from best point
                x1 = best_x
                h = 1e-5 * max(1.0, abs(x1))
                x2 = x1 + h
                try:
                    f1 = call_f(x1)
                    f2 = call_f(x2)
                except Exception:
                    return False
                if abs(f1) <= tol or abs(f2) <= tol:
                    return True
                continue

            if abs(x_new) > 1e15:
                return False

            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False

            # Track stalls
            if abs(f2) < prev_fabs * 0.99:
                prev_fabs = abs(f2)
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 5:
                    # Perturb and restart
                    h = 1e-4 * max(1.0, abs(x2))
                    x2_new = x2 + h
                    try:
                        f2 = call_f(x2_new)
                    except Exception:
                        return False
                    x1, f1 = x2, f2
                    x2 = x2_new
                    prev_fabs = abs(f2)
                    stall_count = 0

        return abs(f1) <= tol or abs(f2) <= tol

    def newton_secant_hybrid(x0, tol=TOL, max_iter=100):
        """
        Hybrid Newton-Secant: uses secant when Newton step is unstable.
        Tracks both a Newton point and a Secant point.
        """
        x = float(x0)
        h = 1e-6 * max(1.0, abs(x))
        x2 = x + h

        try:
            fx = call_f(x)
            fx2 = call_f(x2)
        except Exception:
            return False

        if abs(fx) <= tol:
            return True
        if abs(fx2) <= tol:
            return True

        # Initialize secant points
        x1, f1 = x, fx
        x2, f2 = x2, fx2

        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True

            # Secant step
            denom = f2 - f1
            if abs(denom) < 1e-30:
                h2 = 1e-5 * max(1.0, abs(x2)) * (1 + 0.1 * (i % 10))
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False

            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new) or abs(x_new) > 1e15:
                return False

            # Newton estimate using secant derivative
            d_est = denom / (x2 - x1) if abs(x2 - x1) > 1e-30 else 0
            if abs(d_est) > 1e-30:
                x_newton = x2 - f2 / d_est
                # Use Newton step if it's smaller (more conservative)
                if math.isfinite(x_newton) and abs(x_newton) < abs(x_new) and abs(x_newton) < 1e15:
                    # Blend: take the secant step but check if Newton is closer
                    pass  # Keep secant for stability

            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False

        return abs(f1) <= tol or abs(f2) <= tol

    def bisection_bracket(x0, tol=TOL, max_iter=200):
        """Bisection with efficient bracket search."""
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True

        step = 1e-3 * max(1.0, abs(x))
        a, b, fa, fb = None, None, None, None

        # Search right
        for i in range(30):
            x_test = x + step * (2 ** i)
            try:
                f_test = call_f(x_test)
            except Exception:
                break
            if not math.isfinite(f_test):
                break
            if a is None:
                a, fa = x, fx
            if fa * f_test < 0:
                b, fb = x_test, f_test
                break

        # Search left if not found
        if b is None:
            a, b, fa, fb = None, None, None, None
            for i in range(30):
                x_test = x - step * (2 ** i)
                try:
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
                if b is None:
                    b, fb = x, fx
                if fb * f_test < 0:
                    a, fa = x_test, f_test
                    break

        if a is None or b is None:
            return False

        for i in range(max_iter):
            if not check_time():
                return False
            mid = (a + b) / 2
            try:
                fm = call_f(mid)
            except Exception:
                return False
            if abs(fm) <= tol:
                return True
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break

        return min(abs(fa), abs(fb)) <= tol

    def nelder_mead(x0, tol=TOL, max_iter=200):
        """Nelder-Mead to minimize |f(x)|."""
        x = float(x0)
        scale = 1e-3 * max(1.0, abs(x))
        x_a, x_b = x, x + scale

        try:
            fa = abs(call_f(x_a))
            fb = abs(call_f(x_b))
        except Exception:
            return False

        if fa <= tol or fb <= tol:
            return True

        for i in range(max_iter):
            if not check_time():
                return False

            if fa > fb:
                x_a, x_b = x_b, x_a
                fa, fb = fb, fa

            if fa <= tol:
                return True

            # Reflection
            x_r = x_a + (x_a - x_b)
            try:
                fr = abs(call_f(x_r))
            except Exception:
                break

            if fr <= tol:
                return True

            if fr < fa:
                x_e = x_a + 2.0 * (x_a - x_b)
                try:
                    fe = abs(call_f(x_e))
                except Exception:
                    break
                if fe <= tol:
                    return True
                if fe < fr:
                    x_a, x_b = x_e, x_a
                    fa, fb = fe, fr
                else:
                    x_b = x_a
                    x_a = x_r
                    fa, fb = fr, fa
            elif fr < fb:
                x_b, fb = x_r, fr
            else:
                x_c = x_a + 0.5 * (x_b - x_a)
                try:
                    fc = abs(call_f(x_c))
                except Exception:
                    break
                if fc <= tol:
                    return True
                if fc < fb:
                    x_b, fb = x_c, fc
                else:
                    x_b = x_a + 0.5 * (x_b - x_a)
                    try:
                        fb = abs(call_f(x_b))
                    except Exception:
                        break
                    if abs(x_b - x_a) < 1e-12 * max(1.0, abs(x_a)):
                        new_scale = 1e-2 * max(1.0, abs(x_a))
                        x_b = x_a + new_scale
                        try:
                            fb = abs(call_f(x_b))
                        except Exception:
                            break

            if abs(x_b - x_a) < 1e-14 * max(1.0, abs(x_a)):
                break

        return fa <= tol or fb <= tol

    def polish(x_start, tol=TOL, max_iter=40):
        """High-precision secant polish from a near-solution point."""
        x1 = float(x_start)
        h = 1e-8 * max(1.0, abs(x1))
        x2 = x1 + h

        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False

        if abs(f1) <= tol or abs(f2) <= tol:
            return True

        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol or abs(f1) <= tol:
                return True

            denom = f2 - f1
            if abs(denom) < 1e-30:
                h2 = 1e-8 * max(1.0, abs(x2)) * (1 + 0.01 * i)
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False

            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new) or abs(x_new) > 1e15:
                return False

            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False

        return abs(f1) <= tol or abs(f2) <= tol

    # Strategy cascade
    # 1. Secant with restarts
    if secant_with_restart(x0):
        return best_x

    # 2. Newton-Secant hybrid
    if newton_secant_hybrid(x0):
        return best_x

    # 3. Bisection if budget allows
    if eval_count < max_evals - 100:
        if bisection_bracket(x0):
            return best_x

    # 4. Nelder-Mead minimization
    if nelder_mead(x0):
        return best_x

    # 5. Polish on best point
    if best_fabs < TOL * 1000 and best_fabs > TOL:
        polish(best_x)

    return best_x
# EVOLVE-BLOCK-END
