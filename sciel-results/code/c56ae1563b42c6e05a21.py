"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Scalar root finder for smooth polynomial equations.
    Single Brent-style IQUO (inverse quadratic interpolation) loop with
    bisection fallback.  No derivative estimates, no separate phases.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    TOL = 1e-9
    MAX_CALLS = 490

    x = float(x0)
    call_count = 0
    best_x = x
    best_f = float('inf')

    def _check_time():
        return time.time() < deadline

    def _safe_f(x_val):
        nonlocal call_count, best_x, best_f
        if call_count >= MAX_CALLS:
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            if abs(val) < abs(best_f):
                best_f = val
                best_x = x_val
            return val
        except Exception:
            call_count += 1
            return None

    # --- Phase A: quick bracket search from x0 ---
    fx = _safe_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Try a few candidate offsets to find a sign change
    a, b, fa, fb = None, None, None, None
    step = max(0.1, abs(x) * 0.1)
    for _ in range(30):
        if not _check_time() or call_count >= MAX_CALLS - 5:
            break

        # right
        xr = x + step
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx * fr < 0:
                a, b, fa, fb = x, xr, fx, fr
                break

        # left
        xl = x - step
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx * fl < 0:
                a, b, fa, fb = xl, x, fl, fx
                break

        step *= 2.0
        if step > 1e10:
            break

    if a is None:
        # No bracket found – fall back to best-so-far
        return best_x

    # Ensure fa*fb < 0 (a is the "low" sign, b is the "high" sign)
    if fa * fb > 0:
        a, b = b, a
        fa, fb = fb, fa

    # --- Phase B: Brent-style IQUO loop ---
    # Track three points: a (bracket low), b (bracket high), c (previous best)
    # c starts as a copy of a for the first IQUO attempt.
    c, fc = a, fa

    for _ in range(MAX_CALLS):
        if not _check_time() or call_count >= MAX_CALLS - 2:
            break

        # Check convergence
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        if abs(fc) <= TOL:
            return c

        # Try IQUO (inverse quadratic interpolation) if we have 3 distinct points
        s = None
        if a != b and b != c and a != c:
            da = b - a
            dc = c - a
            db = b - c
            if abs(da) > 1e-15 and abs(dc) > 1e-15 and abs(db) > 1e-15:
                # Standard IQUO formula (Brent):
                s = a + 0.5 * da * (da * fa / (dc * fb) - dc * fc / (db * fb))
                # Only accept if inside bracket
                if not (a < s < b):
                    s = None

        # Fall back to bisection
        if s is None:
            s = (a + b) / 2.0

        fs = _safe_f(s)
        if fs is None:
            break

        if abs(fs) <= TOL:
            return s

        # Update bracket
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        # Update c (previous best) for next IQUO attempt
        if abs(fs) < abs(fc):
            c, fc = s, fs

    # Return the best point found
    candidates = [(abs(fa), a), (abs(fb), b), (abs(fc), c)]
    candidates.sort()
    return candidates[0][1]
# EVOLVE-BLOCK-END
