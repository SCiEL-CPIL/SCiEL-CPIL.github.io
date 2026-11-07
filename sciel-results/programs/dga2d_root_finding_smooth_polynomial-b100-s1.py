"""Root finder for smooth polynomial equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    x = float(x0)

    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # Best point tracking
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # History for interpolation: list of (x, fx), keep up to 3
    history = [(x, fx)]

    # Parameters
    max_iter = 60
    h_fd = 1e-5  # Finite difference step

    def _quadratic_interp_root(x0, f0, x1, f1, x2, f2):
        """Fit quadratic through 3 points, return root closest to x1 (middle point)."""
        # Lagrange interpolation coefficients for the quadratic
        # p(t) = f0*L0(t) + f1*L1(t) + f2*L2(t)
        # where L0, L1, L2 are Lagrange basis polynomials.
        # The quadratic is: a*t^2 + b*t + c
        # Using divided differences:
        d01 = (f1 - f0) / (x1 - x0)
        d12 = (f2 - f1) / (x2 - x1)
        d012 = (d12 - d01) / (x2 - x0)
        # Quadratic: c + d01*(t-x0) + d012*(t-x0)*(t-x1)
        # = d012*t^2 + (d01 - d012*(x0+x1))*t + (f0 - d01*x0 + d012*x0*x1)
        a = d012
        b = d01 - d012 * (x0 + x1)
        c = f0 - d01 * x0 + d012 * x0 * x1
        if abs(a) < 1e-15:
            # Degenerate to linear
            if abs(b) < 1e-15:
                return None
            return -c / b
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        r1 = (-b + sq) / (2 * a)
        r2 = (-b - sq) / (2 * a)
        # Return the root closest to x1 (the most recent/middle point)
        if abs(r1 - x1) <= abs(r2 - x1):
            return r1
        return r2

    for i in range(max_iter):
        if time.time() > deadline:
            break

        x_new = None

        # Priority 1: Quadratic interpolation if we have 3 distinct points
        if len(history) >= 3:
            (xa, fa), (xb, fb), (xc, fc) = history[-3], history[-2], history[-1]
            # Ensure points are distinct enough
            if (abs(xa - xb) > 1e-12 and abs(xb - xc) > 1e-12 and abs(xa - xc) > 1e-12):
                x_new = _quadratic_interp_root(xa, fa, xb, fb, xc, fc)
                if x_new is not None and not math.isfinite(x_new):
                    x_new = None

        # Priority 2: Secant if we have 2 points
        if x_new is None and len(history) >= 2:
            x1, f1 = history[-1]
            x0_h, f0_h = history[-2]
            if abs(f1 - f0_h) > 1e-14 and abs(x1 - x0_h) > 1e-14:
                x_new = x1 - f1 * (x1 - x0_h) / (f1 - f0_h)
                if x_new is not None and not math.isfinite(x_new):
                    x_new = None

        # Priority 3: Newton with FD derivative
        if x_new is None:
            x_h = x + h_fd
            try:
                f_h = f(x_h)
            except Exception:
                x_h = x + 1e-8
                try:
                    f_h = f(x_h)
                except Exception:
                    break
                if not math.isfinite(f_h):
                    break

            if math.isfinite(f_h):
                f_prime = (f_h - fx) / h_fd
                if math.isfinite(f_prime) and abs(f_prime) > 1e-14:
                    x_new = x - fx / f_prime
                    if x_new is not None and not math.isfinite(x_new):
                        x_new = None

        # Fallback: small step
        if x_new is None:
            x_new = x + 1e-3

        # Safeguard: Limit step size
        max_step = max(1.0, abs(x) * 0.5)
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)

        # Evaluate new point
        try:
            f_new = f(x_new)
        except Exception:
            break
        if not math.isfinite(f_new):
            # Shrink step
            x_new = x + 0.5 * (x - x0)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        # Acceptance: |f_new| < |f|
        if abs(f_new) < abs(fx):
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)

            if abs(f_new) <= 1e-8:
                return x_new

            # Update history
            history.append((x_new, f_new))
            if len(history) > 3:
                history.pop(0)

            x = x_new
            fx = f_new
        else:
            # Backtrack: try half step
            x_try = x + 0.5 * (x_new - x)
            try:
                f_try = f(x_try)
            except Exception:
                break
            if math.isfinite(f_try) and abs(f_try) < abs(fx):
                if abs(f_try) < best_abs_fx:
                    best_x = x_try
                    best_fx = f_try
                    best_abs_fx = abs(f_try)
                if abs(f_try) <= 1e-8:
                    return x_try
                history.append((x_try, f_try))
                if len(history) > 3:
                    history.pop(0)
                x = x_try
                fx = f_try
            else:
                # Try a small perturbation to escape
                for dx in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2]:
                    x_pert = x + dx
                    try:
                        f_pert = f(x_pert)
                    except Exception:
                        continue
                    if math.isfinite(f_pert) and abs(f_pert) < abs(fx):
                        if abs(f_pert) < best_abs_fx:
                            best_x = x_pert
                            best_fx = f_pert
                            best_abs_fx = abs(f_pert)
                        if abs(f_pert) <= 1e-8:
                            return x_pert
                        history.append((x_pert, f_pert))
                        if len(history) > 3:
                            history.pop(0)
                        x = x_pert
                        fx = f_pert
                        break
                else:
                    break

    # Final polish: if we're close, do a few more iterations with tighter tolerance
    if abs(best_fx) <= 1e-6:
        x = best_x
        fx = best_fx
        # Use secant/Newton for polish
        for _ in range(20):
            if time.time() > deadline:
                break
            if abs(fx) <= 1e-8:
                break
            # Try Newton with FD
            x_h = x + 1e-7
            try:
                f_h = f(x_h)
            except Exception:
                break
            if not math.isfinite(f_h):
                break
            f_prime = (f_h - fx) / 1e-7
            if math.isfinite(f_prime) and abs(f_prime) > 1e-14:
                x_new = x - fx / f_prime
                if math.isfinite(x_new):
                    try:
                        f_new = f(x_new)
                    except Exception:
                        break
                    if math.isfinite(f_new) and abs(f_new) < abs(fx):
                        x = x_new
                        fx = f_new
                        if abs(fx) < best_abs_fx:
                            best_x = x
                            best_fx = fx
                            best_abs_fx = abs(fx)
                    else:
                        break
            else:
                break
        if abs(best_fx) <= 1e-8:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
