"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder for smooth polynomial equations.
    Strategy:
      1. Quick Newton (forward-diff) to exploit quadratic convergence when close.
      2. If Newton fails/stalls, bracket the root via adaptive expansion.
      3. Solve within bracket using Illinois/modified bisection (fast, robust).
    Tracks best |f(x)| seen and returns best if deadline approached.
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_ok():
        return (time.time() - start_time) < DEADLINE

    x = float(x0)
    if not math.isfinite(x):
        x = 0.0

    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 1: Newton with forward finite-difference derivative ---
    # Use forward difference to save calls (2 calls per step: f(x), f(x+h))
    # but switch to central if needed for accuracy. Start with forward for speed.
    x_newton = x
    fx_newton = best_fx

    for i in range(25):
        if not time_ok():
            break
        if best_abs <= 1e-8:
            return best_x
        if not math.isfinite(fx_newton):
            break

        # Adaptive step size for derivative
        h = 1e-6 * max(1.0, abs(x_newton))
        h = max(h, 1e-12)

        fx_plus = f(x_newton + h)
        if not math.isfinite(fx_plus):
            break

        d = (fx_plus - fx_newton) / h
        if d == 0 or not math.isfinite(d):
            break

        x_candidate = x_newton - fx_newton / d

        # Safeguard against explosion
        if not math.isfinite(x_candidate) or abs(x_candidate) > 1e12:
            break

        fx_candidate = f(x_candidate)
        if not math.isfinite(fx_candidate):
            break

        abs_candidate = abs(fx_candidate)
        if abs_candidate < best_abs:
            best_x = x_candidate
            best_fx = fx_candidate
            best_abs = abs_candidate

        if best_abs <= 1e-12:
            return x_candidate

        # Damping: if step is too large relative to current x, damp it
        step_size = abs(x_candidate - x_newton)
        if step_size > 10 * max(1.0, abs(x_newton)):
            x_newton = x_newton + 0.5 * (x_candidate - x_newton)
            # Re-evaluate
            fx_newton = f(x_newton)
            if not math.isfinite(fx_newton):
                break
            abs_new = abs(fx_newton)
            if abs_new < best_abs:
                best_x = x_newton
                best_fx = fx_newton
                best_abs = abs_new
        else:
            x_newton = x_candidate
            fx_newton = fx_candidate

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 2: Bracket the root ---
    # Try to find [a, b] such that f(a) * f(b) < 0
    a = best_x
    fa = best_fx

    found_bracket = False
    # Try expanding in both directions from best_x and x0
    for start_pt in [best_x, x0]:
        if not time_ok():
            break
        fs = f(start_pt) if start_pt != best_x else fa
        if not math.isfinite(fs):
            continue

        for direction in [1, -1]:
            if not time_ok():
                break
            step = 1.0
            for _ in range(40):
                if not time_ok():
                    break
                b = start_pt + direction * step
                fb = f(b)
                if not math.isfinite(fb):
                    step *= 2
                    if abs(b) > 1e15:
                        break
                    continue

                if fs * fb < 0:
                    # Found bracket
                    if fs * fb < 0:
                        if abs(fs) < abs(fb):
                            a, b = start_pt, b
                            fa, fb = fs, fb
                        else:
                            a, b = b, start_pt
                            fa, fb = fb, fs
                    found_bracket = True
                    break

                step *= 2
                if abs(b) > 1e15:
                    break
            if found_bracket:
                break
        if found_bracket:
            break

    if found_bracket and time_ok():
        # --- Phase 3: Illinois method (modified bisection) ---
        lo, hi = a, b
        flo, fhi = fa, fb

        # Ensure flo and fhi have opposite signs
        if flo * fhi > 0:
            # Shouldn't happen, but just in case
            pass

        for _ in range(100):
            if not time_ok():
                break

            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if not math.isfinite(fmid):
                break

            if abs(fmid) < best_abs:
                best_x = mid
                best_fx = fmid
                best_abs = abs(fmid)

            if best_abs <= 1e-12:
                return mid

            if best_abs <= 1e-8:
                return best_x

            # Illinois: adjust the side with larger |f|
            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid

            # Illinois damping: shrink the interval on the side with larger |f|
            if abs(flo) > abs(fhi):
                # Swap roles effectively by damping flo
                lo = 0.5 * (lo + hi)
                flo = f(lo)
                if not math.isfinite(flo):
                    break
            else:
                hi = 0.5 * (lo + hi)
                fhi = f(hi)
                if not math.isfinite(fhi):
                    break

            if abs(hi - lo) < 1e-14 * max(1.0, abs(mid)):
                break

        if best_abs <= 1e-8:
            return best_x

    # --- Phase 4: Secant method as fallback ---
    # Use two points: x0 and x0 + small delta
    x1 = float(x0)
    x2 = x1 + 1e-5
    if not time_ok():
        return best_x

    f1 = f(x1)
    f2 = f(x2)

    if math.isfinite(f1) and math.isfinite(f2) and abs(f2 - f1) > 1e-30:
        for _ in range(50):
            if not time_ok():
                break
            if abs(f2 - f1) < 1e-30:
                break
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x3) or abs(x3) > 1e12:
                break

            f3 = f(x3)
            if not math.isfinite(f3):
                break

            if abs(f3) < best_abs:
                best_x = x3
                best_fx = f3
                best_abs = abs(f3)

            if best_abs <= 1e-12:
                return x3
            if best_abs <= 1e-8:
                return best_x

            x1, f1 = x2, f2
            x2, f2 = x3, f3

            if abs(x2 - x1) < 1e-14 * max(1.0, abs(x2)):
                break

    return best_x
# EVOLVE-BLOCK-END
