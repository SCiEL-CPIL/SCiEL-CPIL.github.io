"""Root finder: derivative-free Brent variant with bracketing, IQI, bisection, deadline guard."""
import math
import time

# Safety margin: stop new work this far before the 150s wall-clock limit.
DEADLINE_MARGIN = 5.0
MAX_CALLS = 490
TOL = 1e-8
TOL_POLISH = 1e-12


def _iqi(x1, f1, x2, f2, x3, f3):
    """Inverse quadratic interpolation node from three points."""
    d1 = (x3 - x1) * f2
    d2 = (x3 - x2) * f1
    d3 = (x2 - x1) * f3
    num = d1 * d2 + d2 * d3 + d1 * d3
    den = (x3 - x1) * (x3 - x2) * f1 + (x3 - x2) * (x3 - x1) * f2 + (x2 - x1) * (x3 - x1) * f3
    if den == 0.0:
        return None
    return num / den


def find_root(f, x0):
    start = time.time()

    state = {"calls": 0, "best": (None, None), "points": []}  # points: list of (x, fx)

    def eval_f(x):
        fx = f(x)
        state["calls"] += 1
        if math.isfinite(fx):
            afx = abs(fx)
            if state["best"][0] is None or afx < state["best"][1]:
                state["best"] = (x, afx)
            state["points"].append((x, fx))
            if len(state["points"]) > 8:
                state["points"].pop(0)
        return fx

    def time_ok():
        return (time.time() - start) < (150.0 - DEADLINE_MARGIN)

    def calls_ok():
        return state["calls"] < MAX_CALLS

    def best_x():
        bx, _ = state["best"]
        if bx is not None:
            return bx
        return float(x0)

    x = float(x0)
    fx = eval_f(x)
    if math.isfinite(fx) and abs(fx) <= TOL:
        return x

    # ---- Phase 1: find a bracket [a, b] with a sign change ----
    a = x
    fa = fx
    b = None
    fb = None

    # Candidate step sizes: try small nearby first, then expand.
    steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0,
             100.0, 500.0, 1000.0, 5000.0, 10000.0]

    for delta in steps:
        if not (time_ok() and calls_ok()):
            break
        for sign in (1, -1):
            if not (time_ok() and calls_ok()):
                break
            xn = x + sign * delta
            fn = eval_f(xn)
            if not math.isfinite(fn):
                continue
            if abs(fn) <= TOL:
                return xn
            if fa * fn < 0:
                a, fa = x, fx
                b, fb = xn, fn
                break
        if b is not None:
            break

    # If no bracket found, fall back to a local minimization of |f| via golden section
    # on a wide interval around x0 to try to reduce |f|, then return best.
    if b is None:
        if time_ok() and calls_ok():
            # Golden-section search minimizing |f| on [x0 - R, x0 + R]
            R = max(1.0, abs(x0)) * 10.0
            lo, hi = x0 - R, x0 + R
            invphi = (math.sqrt(5.0) - 1.0) / 2.0
            c = hi - invphi * (hi - lo)
            d = lo + invphi * (hi - lo)
            fc = abs(eval_f(c))
            fd = abs(eval_f(d))
            for _ in range(60):
                if not (time_ok() and calls_ok()):
                    break
                if fc < fd:
                    hi = d
                    d = c
                    fd = fc
                    c = hi - invphi * (hi - lo)
                    fc = abs(eval_f(c))
                else:
                    lo = c
                    c = d
                    fc = fd
                    d = lo + invphi * (hi - lo)
                    fd = abs(eval_f(d))
        return best_x()

    # ---- Phase 2: Brent's method on [a, b] ----
    # Maintain c as the previous b, and use IQI when safe.
    c = b
    fc = fb
    # We have a, fa and b, fb with fa*fb < 0.
    # Keep the bracket such that |fb| <= |fa| to bias toward the closer end.
    if abs(fb) > abs(fa):
        a, b = b, a
        fa, fb = fb, fa
    c = b
    fc = fb

    for _ in range(200):
        if not (time_ok() and calls_ok()):
            break
        if math.isfinite(fb) and abs(fb) <= TOL:
            return b
        if math.isfinite(fa) and abs(fa) <= TOL:
            return a

        # Inverse quadratic interpolation
        x_iqi = None
        # Use the three most recent distinct points for IQI: a, b, c
        # Ensure distinctness
        if a != b and b != c and a != c:
            cand = _iqi(a, fa, b, fb, c, fc)
            if cand is not None and math.isfinite(cand):
                # Safety: candidate must be strictly inside the bracket
                lo_br, hi_br = (a, b) if a < b else (b, a)
                if lo_br < cand < hi_br:
                    # Additional "safe" criterion: not too close to midpoint
                    mid = (a + b) / 2.0
                    x_iqi = cand

        if x_iqi is None:
            # Bisection
            x_iqi = (a + b) / 2.0
            if x_iqi == a or x_iqi == b:
                x_iqi = (a + b) / 2.0

        # Evaluate
        fxn = eval_f(x_iqi)
        if not math.isfinite(fxn):
            # Fallback: bisect
            x_iqi = (a + b) / 2.0
            if x_iqi == a or x_iqi == b:
                x_iqi = (a + b) / 2.0
            fxn = eval_f(x_iqi)
            if not math.isfinite(fxn):
                break

        if abs(fxn) <= TOL:
            return x_iqi

        # Update bracket: keep the subinterval with a sign change
        if fa * fxn < 0:
            b, fb = a, fa
        else:
            a, fa = x_iqi, fxn

        # Keep |fb| <= |fa| and update c
        if abs(fb) > abs(fa):
            a, b = b, a
            fa, fb = fb, fa
        c = b
        fc = fb

        # Convergence by interval width
        if abs(a - b) < 1e-14:
            break

    # ---- Phase 3: final high-precision polish (few extra calls, tighter tol) ----
    if time_ok() and calls_ok():
        # Polish around the best point using a few secant/IQI steps with tighter tol
        bx, _ = state["best"]
        if bx is not None and math.isfinite(bx):
            px = bx
            pfx = eval_f(px)
            if math.isfinite(pfx) and abs(pfx) <= TOL:
                return px
            # A few secant steps with tighter tolerance
            for _ in range(10):
                if not (time_ok() and calls_ok()):
                    break
                if abs(pfx) <= TOL_POLISH:
                    return px
                # Use the previous point in state.points for secant if available
                prev = None
                for (xx, fxx) in reversed(state["points"]):
                    if xx != px and math.isfinite(fxx):
                        prev = (xx, fxx)
                        break
                if prev is None:
                    break
                xx, fxx = prev
                if fxx == pfx:
                    break
                step = (xx - px) * pfx / (fxx - pfx)
                nx = px + step
                if not math.isfinite(nx):
                    break
                nfx = eval_f(nx)
                if not math.isfinite(nfx):
                    break
                px, pfx = nx, nfx
                if abs(pfx) <= TOL:
                    return px

    return best_x()
