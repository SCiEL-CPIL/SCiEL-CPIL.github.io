"""Root finder for 3 fixed smooth polynomial equations; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def _is_finite(x):
    return x is not None and math.isfinite(x)

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a hybrid bracketing + safeguarded
    secant with Newton polish.  Structural change: a clean two-phase design
    (bracket first, then converge) with call-reuse in the polish phase.
    """
    deadline = time.monotonic() + 105.0  # safety margin before 120 s kill
    MAX_CALLS = 450
    TOL = 1e-10  # slightly tighter than 1e-8 for safety
    calls = 0

    def g(x):
        nonlocal calls
        if calls >= MAX_CALLS:
            return None
        try:
            y = f(x)
        except Exception:
            return None
        if not _is_finite(y):
            return None
        calls += 1
        return float(y)

    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    # --- Phase 1: initial evaluation ---
    fx0 = g(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f = fx0
    best_abs = abs(fx0)

    def consider(x, fx):
        nonlocal best_x, best_f, best_abs
        if fx is None or not _is_finite(x):
            return
        a = abs(fx)
        if a < best_abs:
            best_abs = a
            best_x = x
            best_f = fx

    # --- Phase 2: bracket search ---
    # Try small steps first, then geometric expansion in both directions.
    a = b = fa = fb = None

    # Small perturbations
    h0 = max(1e-4, abs(x0) * 1e-4)
    for sign in (1.0, -1.0):
        trial = x0 + sign * h0
        ft = g(trial)
        consider(trial, ft)
        if ft is None:
            continue
        if abs(ft) <= TOL:
            return trial
        if fx0 * ft < 0.0:
            if fx0 < 0:
                a, b = x0, trial
                fa, fb = fx0, ft
            else:
                a, b = trial, x0
                fa, fb = ft, fx0
            break

    # Geometric expansion
    if a is None:
        base = max(1.0, abs(x0))
        for mult in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
                     256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0,
                     32768.0, 65536.0, 131072.0, 262144.0):
            if time.monotonic() > deadline:
                break
            d = base * mult
            for sign in (1.0, -1.0):
                trial = x0 + sign * d
                ft = g(trial)
                consider(trial, ft)
                if ft is None:
                    continue
                if abs(ft) <= TOL:
                    return trial
                if fx0 * ft < 0.0:
                    if fx0 < 0:
                        a, b = x0, trial
                        fa, fb = fx0, ft
                    else:
                        a, b = trial, x0
                        fa, fb = ft, fx0
                    break
            if a is not None:
                break

    # --- Phase 3: convergence ---
    if a is not None and b is not None:
        # Safeguarded secant with bisection fallback
        lo, hi = (a, b) if a < b else (b, a)
        f_lo, f_hi = (fa, fb) if a < b else (fb, fa)
        x_prev = lo
        f_prev = f_lo
        x_curr = hi
        f_curr = f_hi

        for _ in range(200):
            if time.monotonic() > deadline:
                break
            if calls >= MAX_CALLS:
                break
            if abs(f_curr) <= TOL:
                return x_curr
            if abs(f_prev) <= TOL:
                return x_prev

            # Secant prediction
            denom = f_curr - f_prev
            if denom != 0.0:
                x_sec = x_curr - f_curr * (x_curr - x_prev) / denom
            else:
                x_sec = 0.5 * (lo + hi)

            if not _is_finite(x_sec):
                x_sec = 0.5 * (lo + hi)

            # Safeguard
            if x_sec <= lo or x_sec >= hi:
                x_new = 0.5 * (lo + hi)
            else:
                x_new = x_sec

            f_new = g(x_new)
            consider(x_new, f_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new

            # Update bracket
            if f_new * f_lo <= 0.0:
                hi = x_new
                f_hi = f_new
            else:
                lo = x_new
                f_lo = f_new

            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new

            # Convergence on interval
            if abs(hi - lo) < 1e-14 * max(1.0, abs(lo), abs(hi)):
                break

        # Return best endpoint
        if abs(f_lo) <= abs(f_hi):
            return lo
        return hi

    # --- No bracket found: secant method with damping ---
    xa = x0
    fa_val = fx0
    h = max(1e-4, abs(xa) * 1e-4)
    xb = xa + h
    fb_val = g(xb)
    consider(xb, fb_val)
    if fb_val is None:
        return best_x
    if abs(fb_val) <= TOL:
        return xb

    ref_x = best_x
    ref_f = best_f
    for _ in range(200):
        if time.monotonic() > deadline:
            break
        if calls >= MAX_CALLS:
            break
        if abs(fb_val) <= TOL:
            return xb
        if abs(fa_val) <= TOL:
            return xa

        denom = fb_val - fa_val
        if denom != 0.0:
            s = xb - fb_val * (xb - xa) / denom
        else:
            s = (xa + xb) / 2.0

        if not _is_finite(s):
            s = (xa + xb) / 2.0

        # Damp if step is too large
        max_step = max(1.0, abs(ref_x)) * 1e6
        if abs(s - ref_x) > max_step:
            s = ref_x + math.copysign(max_step, s - ref_x)

        fs = g(s)
        consider(s, fs)
        if fs is None:
            break
        if abs(fs) <= TOL:
            return s

        if abs(fs) < abs(fa_val):
            xa, fa_val = xb, fb_val
            xb, fb_val = s, fs
        else:
            xa, fa_val = s, fs

        if abs(fs) < abs(ref_f):
            ref_x, ref_f = s, fs

    # --- Phase 4: Newton polish with central difference (call-reuse) ---
    px = best_x
    pf = best_f
    if abs(pf) > 1e-8:
        # Pick the point with smallest |f|
        if abs(fa_val) < abs(pf):
            px, pf = xa, fa_val
        elif abs(fb_val) < abs(pf):
            px, pf = xb, fb_val

    for _ in range(12):
        if time.monotonic() > deadline:
            break
        if calls >= MAX_CALLS:
            break
        if abs(pf) <= 1e-12:
            return px
        h = 1e-7 * max(1.0, abs(px))
        fp = g(px + h)
        fm = g(px - h)
        if fp is None or fm is None:
            break
        d = (fp - fm) / (2.0 * h)
        if d == 0.0 or not _is_finite(d):
            break
        nx = px - pf / d
        if not _is_finite(nx):
            break
        nf = g(nx)
        consider(nx, nf)
        if nf is None:
            break
        if abs(nf) <= 1e-12:
            return nx
        px, pf = nx, nf

    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
