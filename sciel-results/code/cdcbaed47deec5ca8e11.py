"""Root finder for 3 fixed smooth polynomial equations; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def _is_finite(x):
    return x is not None and math.isfinite(x)

def _safe_f(f, x):
    try:
        y = f(x)
    except Exception:
        return None
    if not _is_finite(y):
        return None
    return float(y)

def _secant_step(a, fa, b, fb):
    # Secant (inverse interpolation) step from the two endpoints.
    denom = fb - fa
    if denom == 0.0:
        return (a + b) / 2.0
    s = b - (fb * (b - a)) / denom
    return s

def _bisect_step(a, fa, b, fb):
    # Bisection step, choosing the half-interval that still brackets a sign change.
    m = (a + b) / 2.0
    fm = _safe_f(f, m) if 'f' in globals() else None
    return m

def find_root(f, x0):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s wall-clock
    MAX_CALLS = 400
    calls = 0
    TOL = 1e-12  # internal target; evaluator needs |f| <= 1e-8

    def g(x):
        nonlocal calls
        if calls >= MAX_CALLS:
            return None
        y = _safe_f(f, x)
        calls += 1
        return y

    # Track best finite point by |f|.
    best_x = None
    best_f = None
    best_abs = math.inf

    def consider(x, fx):
        nonlocal best_x, best_f, best_abs
        if fx is None or not _is_finite(x):
            return
        a = abs(fx)
        if a < best_abs:
            best_abs = a
            best_x = x
            best_f = fx

    # ---- Initial probe & bracketing ----
    x = float(x0)
    fx = g(x)
    consider(x, fx)
    if fx is not None and abs(fx) <= TOL:
        return x

    # Try to build a bracket [a,b] around x0.
    a = b = None
    fa = fb = None
    if fx is not None:
        # Expand outward to find a sign change.
        step = max(1.0, abs(x)) * 0.5
        # Try shrinking first (in case x0 is near root), then expanding.
        for trial in (x + step, x - step):
            ft = g(trial)
            consider(trial, ft)
            if ft is None:
                continue
            if abs(ft) <= TOL:
                return trial
            if fx is not None and fx * ft < 0.0:
                a, b = (x, trial) if fx * ft < 0 else (trial, x)
                fa, fb = (fx, ft) if a == x else (ft, fx)
                break
        if a is None:
            # Expand both directions geometrically.
            base = max(1.0, abs(x))
            for mult in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0):
                d = base * mult
                for sign in (1.0, -1.0):
                    trial = x + sign * d
                    ft = g(trial)
                    consider(trial, ft)
                    if ft is None:
                        continue
                    if abs(ft) <= TOL:
                        return trial
                    if fx is not None and fx * ft < 0.0:
                        a, b = (x, trial) if fx * ft < 0 else (trial, x)
                        fa, fb = (fx, ft) if a == x else (ft, fx)
                        break
                if a is not None:
                    break

    # ---- Main loop: safeguarded secant/bisection ----
    if a is not None and b is not None:
        # We have a bracket. Use inverse-interpolation (secant) with bisection safeguard.
        for _ in range(300):
            if time.time() > deadline:
                break
            if calls >= MAX_CALLS:
                break
            if abs(fb) <= TOL:
                return b
            if abs(fa) <= TOL:
                return a
            # Secant candidate.
            s = _secant_step(a, fa, b, fb)
            # Safeguard: s must lie strictly inside (a,b) and not too close to endpoints.
            lo, hi = (a, b) if a < b else (b, a)
            if not (lo < s < hi):
                s = (a + b) / 2.0
            fs = g(s)
            consider(s, fs)
            if fs is None:
                break
            if abs(fs) <= TOL:
                return s
            # Update bracket to the subinterval that brackets a sign change.
            if fa * fs < 0.0:
                b, fb = s, fs
            elif fb * fs < 0.0:
                a, fa = s, fs
            else:
                # No sign change: take the side with smaller |f| to keep progress.
                if abs(fs) < abs(fa):
                    a, fa = s, fs
                else:
                    b, fb = s, fs
        # Return best bracket endpoint.
        if abs(fa) <= abs(fb):
            return a
        return b

    # ---- No bracket: pure safeguarded secant (Brent-like) with damping ----
    # Maintain two most recent points and step with damping if it leaves a reasonable region.
    xa = float(x0)
    fa = g(xa)
    consider(xa, fa)
    if fa is not None and abs(fa) <= TOL:
        return xa
    # Second point.
    h = max(1e-4, abs(xa) * 1e-4)
    xb = xa + h
    fb = g(xb)
    consider(xb, fb)
    if fb is not None and abs(fb) <= TOL:
        return xb

    # Track a "safe" reference: the best point so far; damp steps that increase |f| a lot.
    ref_x = best_x
    ref_f = best_f
    for _ in range(300):
        if time.time() > deadline:
            break
        if calls >= MAX_CALLS:
            break
        if fb is None or fa is None:
            break
        if abs(fb) <= TOL:
            return xb
        if abs(fa) <= TOL:
            return xa
        s = _secant_step(xa, fa, xb, fb)
        # Damping: if step is huge or goes far from reference, pull back toward reference.
        if not _is_finite(s):
            s = (xa + xb) / 2.0
        # Limit step magnitude relative to reference.
        if ref_x is not None:
            max_step = max(1.0, abs(ref_x)) * 1e6
            if abs(s - ref_x) > max_step:
                s = ref_x + math.copysign(max_step, s - ref_x)
        fs = g(s)
        consider(s, fs)
        if fs is None:
            break
        if abs(fs) <= TOL:
            return s
        # Update: keep the two points with smaller |f| among (xa, xb, s) to stay near root.
        # Standard secant: replace the older point.
        if abs(fs) < abs(fa):
            # s is better than xa; shift window.
            xa, fa = xb, fb
            xb, fb = s, fs
        else:
            xa, fa = s, fs
        # Update reference if improved.
        if best_x is not None and abs(best_f if best_f is not None else math.inf) < abs(fa) and abs(best_f if best_f is not None else math.inf) < abs(fb):
            pass
        if abs(fs) < abs(ref_f if ref_f is not None else math.inf):
            ref_x, ref_f = s, fs

    # ---- Final high-precision polish on the best point ----
    if best_x is not None:
        # A few Newton-ish steps using secant with the best point and a tiny neighbor.
        px = best_x
        pf = best_f
        for _ in range(8):
            if time.time() > deadline:
                break
            if calls >= MAX_CALLS:
                break
            if abs(pf) <= 1e-14:
                return px
            hh = 1e-6 * max(1.0, abs(px))
            qx = px + hh
            qf = g(qx)
            consider(qx, qf)
            if qf is None:
                break
            denom = qf - pf
            if denom == 0.0:
                break
            nx = px - pf * (qx - px) / denom
            if not _is_finite(nx):
                break
            nf = g(nx)
            consider(nx, nf)
            if nf is None:
                break
            if abs(nf) <= 1e-14:
                return nx
            px, pf = nx, nf

    # Return the best finite point found.
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
