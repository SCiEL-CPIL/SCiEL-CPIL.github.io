"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder for smooth polynomial equations.
    Strategy:
      1. Secant method (1 call/step after initial) for smooth polynomials.
      2. If secant stalls, bracket via adaptive expansion.
      3. Solve within bracket using Illinois method (faster than pure bisection).
    Tracks best |f(x)| seen and returns best if deadline approached.
    """
    start_time = time.time()
    DEADLINE = 148.0  # safety margin before 150s kill

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

    # --- Phase 1: Secant method ---
    # Use x0 and x0 + small delta as initial points
    x1 = x
    f1 = best_fx
    x2 = x1 + 1e-4 * max(1.0, abs(x1))
    f2 = f(x2)

    if not math.isfinite(f2):
        x2 = x1 - 1e-4 * max(1.0, abs(x1))
        f2 = f(x2)

    if math.isfinite(f1) and math.isfinite(f2) and abs(f2 - f1) > 1e-30:
        for _ in range(50):
            if not time_ok():
                break
            if best_abs <= 1e-8:
                return best_x
            if abs(f2 - f1) < 1e-30:
                break

            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x3) or abs(x3) > 1e15:
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

            # Check convergence
            if abs(x3 - x2) < 1e-14 * max(1.0, abs(x3)):
                break

            x1, f1 = x2, f2
            x2, f2 = x3, f3

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 2: Bracket the root ---
    found_bracket = False
    # Try expanding in both directions from best_x first, then x0
    for start_pt in [best_x, x0]:
        if not time_ok():
            break
        fs = f(start_pt) if start_pt != best_x else best_fx
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
        # --- Phase 3: Illinois method (modified bisection, 1-2 calls/step) ---
        lo, hi = a, b
        flo, fhi = fa, fb

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

    # --- Phase 4: Secant restart from best_x with perturbation ---
    if time_ok():
        x1 = best_x
        f1 = best_fx
        x2 = x1 + 1e-3 * max(1.0, abs(x1))
        f2 = f(x2)

        if math.isfinite(f1) and math.isfinite(f2) and abs(f2 - f1) > 1e-30:
            for _ in range(30):
                if not time_ok():
                    break
                if best_abs <= 1e-8:
                    return best_x
                if abs(f2 - f1) < 1e-30:
                    break

                x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
                if not math.isfinite(x3) or abs(x3) > 1e15:
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

                if abs(x3 - x2) < 1e-14 * max(1.0, abs(x3)):
                    break

                x1, f1 = x2, f2
                x2, f2 = x3, f3

    return best_x
# EVOLVE-BLOCK-END
