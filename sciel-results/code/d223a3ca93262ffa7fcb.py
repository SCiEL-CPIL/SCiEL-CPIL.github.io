"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses a combination of:
    1. Newton's method with adaptive damping and line search
    2. Secant method fallback
    3. Bisection for guaranteed convergence when bracketed
    4. Multiple restarts if initial guess is poor
    5. Deadline guard to stay within time limits
    """
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    max_evals = 480
    eval_count = [0]
    best_x = float(x0)
    best_fx = None

    def eval_f(x):
        nonlocal best_x, best_fx
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if val is not None and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                if best_fx is None or abs(val) < abs(best_fx):
                    best_fx = val
                    best_x = float(x)
            return val
        except (ValueError, OverflowError, ZeroDivisionError, ArithmeticError):
            eval_count[0] += 1
            return None

    def is_finite(x):
        return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

    def newton_step(x, fx):
        """Try one Newton step with damping. Returns (new_x, new_fx) or None."""
        h = 1e-8 * max(1.0, abs(x))
        fx_plus = eval_f(x + h)
        if fx_plus is None:
            return None
        if fx is None:
            fx = eval_f(x)
            if fx is None:
                return None
        d = (fx_plus - fx) / h
        if abs(d) < 1e-15:
            return None
        step = fx / d
        # Damped Newton: try full step, then halve
        for damping in [1.0, 0.5, 0.25, 0.1, 0.01]:
            new_x = x - damping * step
            if not is_finite(new_x):
                continue
            new_fx = eval_f(new_x)
            if new_fx is None:
                continue
            # Accept if |f| decreased
            if abs(new_fx) < abs(fx):
                return (new_x, new_fx)
        return None

    def secant_step(x_prev, fx_prev, x_curr, fx_curr):
        """Secant method step. Returns (new_x, new_fx) or None."""
        if fx_prev is None or fx_curr is None:
            return None
        denom = fx_curr - fx_prev
        if abs(denom) < 1e-15:
            return None
        x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
        if not is_finite(x_new):
            return None
        fx_new = eval_f(x_new)
        if fx_new is None:
            return None
        return (x_new, fx_new)

    def try_newton(x0, max_iters=80):
        """Run Newton's method with damping from x0."""
        x = float(x0)
        fx = eval_f(x)
        if fx is None:
            return None
        if abs(fx) <= 1e-10:
            return x
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            result = newton_step(x, fx)
            if result is None:
                return None
            x, fx = result
            if abs(fx) <= 1e-10:
                return x
        return None

    def try_secant(x0, max_iters=100):
        """Run secant method from x0 with a small perturbation."""
        x0 = float(x0)
        x1 = x0 + 1e-6 * max(1.0, abs(x0))
        fx0 = eval_f(x0)
        fx1 = eval_f(x1)
        if fx0 is None or fx1 is None:
            return None
        if abs(fx0) <= 1e-10:
            return x0
        if abs(fx1) <= 1e-10:
            return x1
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            result = secant_step(x0, fx0, x1, fx1)
            if result is None:
                return None
            x0, fx0 = x1, fx1
            x1, fx1 = result
            if abs(fx1) <= 1e-10:
                return x1
            if abs(fx0) <= 1e-10:
                return x0
        return None

    def try_bisection(a, b, fa, fb, max_iters=200):
        """Bisection method given a bracket [a,b] with sign change."""
        if fa is None or fb is None:
            return None
        if fa * fb > 0:
            return None
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            mid = 0.5 * (a + b)
            fm = eval_f(mid)
            if fm is None:
                return None
            if abs(fm) <= 1e-10:
                return mid
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
                return mid
        return None

    def find_bracket(x0, max_steps=50):
        """Try to find a bracket around x0 by expanding interval."""
        x0 = float(x0)
        fx0 = eval_f(x0)
        if fx0 is None:
            return None
        if abs(fx0) <= 1e-10:
            return (x0, fx0)
        step = 1e-6 * max(1.0, abs(x0))
        for i in range(max_steps):
            if time.time() > deadline:
                return None
            # Try expanding in both directions
            for direction in [1, -1]:
                x_try = x0 + direction * step * (2 ** i)
                fx_try = eval_f(x_try)
                if fx_try is None:
                    continue
                if fx0 * fx_try < 0:
                    return (x_try, fx_try)
        return None

    # Strategy: try multiple methods in sequence, keep best result
    x0 = float(x0)

    # 1. Try Newton first
    result = try_newton(x0)
    if result is not None and best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # 2. Try secant
    result = try_secant(x0)
    if result is not None and best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # 3. Try to find a bracket and use bisection
    bracket = find_bracket(x0)
    if bracket is not None:
        x_br, fx_br = bracket
        fx0 = eval_f(x0)
        if fx0 is not None and fx0 * fx_br < 0:
            result = try_bisection(x0, x_br, fx0, fx_br)
            if result is not None and best_fx is not None and abs(best_fx) <= 1e-8:
                return best_x

    # 4. Try secant with different initial points
    for offset in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2]:
        x_try = x0 + offset * max(1.0, abs(x0))
        result = try_secant(x_try)
        if result is not None and best_fx is not None and abs(best_fx) <= 1e-8:
            return best_x

    # 5. Final Newton attempt from best_x
    if best_fx is not None:
        result = try_newton(best_x)
        if result is not None and best_fx is not None and abs(best_fx) <= 1e-8:
            return best_x

    # Return the best we found
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
