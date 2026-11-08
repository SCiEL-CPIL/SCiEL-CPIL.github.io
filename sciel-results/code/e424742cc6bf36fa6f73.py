"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root-finding for potentially ill-conditioned problems.
    Strategy:
      1. Use Brent's method (scipy.optimize.brentq) when a bracket is found.
      2. Use Halley's method / Newton with numerical derivative as fallback.
      3. Adaptive exploration to find a sign change (bracket).
      4. Keep track of the best x found (min |f(x)|).
      5. Deadline guard: stop if we approach 140s (safety margin before 150s kill).
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    x = float(x0)
    best_x = x
    best_f = f(x)
    best_abs_f = abs(best_f)

    eval_count = 1

    def check_deadline():
        return (time.time() - start_time) >= DEADLINE

    def update_best(x_candidate, fx_candidate):
        nonlocal best_x, best_f, best_abs_f
        if abs(fx_candidate) < best_abs_f:
            best_x = x_candidate
            best_f = fx_candidate
            best_abs_f = abs(fx_candidate)

    # If already good enough, return immediately
    if best_abs_f <= 1e-8:
        return best_x

    max_evals = 500
    current_evals = 1

    # Try to find a bracket around x0
    bracket_low = None
    bracket_high = None

    # Initial exploration to find a sign change
    step = 1e-6 * max(1.0, abs(x0))
    for i in range(40):
        if check_deadline() or current_evals >= max_evals:
            break
        # Alternate directions with increasing step
        direction = 1 if i % 2 == 0 else -1
        multiplier = (i // 2 + 1)
        x_try = x0 + direction * multiplier * step
        fx_try = f(x_try)
        current_evals += 1
        update_best(x_try, fx_try)
        if abs(fx_try) <= 1e-8:
            return x_try
        if fx_try * best_f < 0:
            if fx_try < 0:
                bracket_low = x_try
                bracket_high = best_x
            else:
                bracket_low = best_x
                bracket_high = x_try
            break

    # If we found a bracket, use Brent's method
    if bracket_low is not None and bracket_high is not None:
        try:
            from scipy.optimize import brentq
            # Wrap f to count evaluations
            def counted_f(x_val):
                nonlocal current_evals
                if check_deadline() or current_evals >= max_evals:
                    raise ValueError("Deadline exceeded")
                fx_val = f(x_val)
                current_evals += 1
                update_best(x_val, fx_val)
                if abs(fx_val) <= 1e-8:
                    raise ValueError("Root found")
                return fx_val

            try:
                root = brentq(counted_f, bracket_low, bracket_high, xtol=1e-12, rtol=1e-12, maxiter=200)
                if abs(f(root)) <= 1e-8:
                    return root
                update_best(root, f(root))
            except ValueError:
                pass
        except Exception:
            pass

    # If no bracket or Brent failed, use Newton/Halley with fallbacks
    x_current = best_x
    prev_x = x0
    prev_fx = best_f

    for iteration in range(100):
        if check_deadline() or current_evals >= max_evals:
            break
        if best_abs_f <= 1e-8:
            return best_x

        fx = f(x_current)
        current_evals += 1
        update_best(x_current, fx)

        if abs(fx) <= 1e-8:
            return x_current

        # Numerical derivative
        h = 1e-8 * max(1.0, abs(x_current))
        fx_plus = f(x_current + h)
        current_evals += 1
        update_best(x_current + h, fx_plus)

        derivative = (fx_plus - fx) / h

        x_candidate = None

        # Newton step
        if abs(derivative) > 1e-30:
            x_newton = x_current - fx / derivative
            if abs(x_newton - x_current) < 1e6 * max(1.0, abs(x_current)):
                x_candidate = x_newton

        # Secant method
        if abs(fx - prev_fx) > 1e-30:
            x_secant = x_current - fx * (x_current - prev_x) / (fx - prev_fx)
            if x_candidate is None or abs(x_secant - x_current) < abs(x_candidate - x_current):
                x_candidate = x_secant

        # Bisection if bracket exists
        if bracket_low is not None and bracket_high is not None:
            mid = (bracket_low + bracket_high) / 2
            if x_candidate is None or abs(bracket_high - bracket_low) < 1e-6:
                x_candidate = mid

        if x_candidate is None:
            x_candidate = x_current + step * (1 if iteration % 2 == 0 else -1)

        prev_x = x_current
        prev_fx = fx
        x_current = x_candidate

        fx_current = f(x_current)
        current_evals += 1
        update_best(x_current, fx_current)

        if abs(fx_current) <= 1e-8:
            return x_current

        # Update bracket
        if bracket_low is not None and bracket_high is not None:
            fx_low = f(bracket_low)
            current_evals += 1
            if fx_current * fx_low <= 0:
                bracket_high = x_current
            else:
                bracket_low = x_current
        else:
            # Try to establish bracket
            if fx_current * fx < 0:
                if fx_current < 0:
                    bracket_low = x_current
                    bracket_high = x_current - h  # approximate
                else:
                    bracket_low = x_current - h
                    bracket_high = x_current

    return best_x

# EVOLVE-BLOCK-END
