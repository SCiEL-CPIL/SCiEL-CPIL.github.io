"""Scalar root finder for 3 fixed smooth polynomial equations.

find_root(f, x0) -> float with |f(x)| <= 1e-8, minimizing f-calls.
Backbone: guaranteed bisection with Newton/secant acceleration.
"""
import math
import time

# EVOLVE-BLOCK-START
_TOL = 1e-9          # internal acceptance (margin below the 1e-8 contract)
_MAX_CALLS = 440     # hard per-equation cap (well under the 500 failure limit)
_DEADLINE = 132.0    # wall-clock stop (evaluator kills at 150 s)
_START = time.monotonic()


def _is_bad(v):
    return (v is None) or (not math.isfinite(v))


def _find_bracket(f, x0, counter, budget, deadline):
    """Find an interval [a,b] with a sign change, using few calls.
    Returns (x, fx) if an exact root is hit, else (a, b) with a sign change,
    else None."""
    if _is_bad(x0):
        x0 = 0.0

    x = float(x0)
    fx = f(x)
    counter["n"] += 1
    if abs(fx) <= _TOL:
        return (x, fx)
    if not math.isfinite(fx):
        x = 0.0
        fx = f(x)
        counter["n"] += 1
        if abs(fx) <= _TOL:
            return (x, fx)
        if not math.isfinite(fx):
            return None

    # 1) expand around x0 (geometric, both directions)
    for it in range(45):
        if counter["n"] >= budget or time.monotonic() >= deadline:
            break
        for s in (1.0, -1.0):
            nx = x + s * (1.0 + 2.0 ** it)
            fn = f(nx)
            counter["n"] += 1
            if _is_bad(fn):
                continue
            if abs(fn) <= _TOL:
                return (nx, fn)
            if fn * fx < 0.0:
                return (x, nx)
        if counter["n"] >= budget or time.monotonic() >= deadline:
            break

    # 2) coarse global sweep (odd number of intervals)
    for (m, n) in ((64.0, 129), (160.0, 65), (400.0, 65)):
        lo, hi = -m, m
        step = (hi - lo) / (n - 1)
        prev_x = lo
        prev_f = f(lo)
        counter["n"] += 1
        if _is_bad(prev_f):
            continue
        if abs(prev_f) <= _TOL:
            return (lo, prev_f)
        for k in range(1, n):
            if counter["n"] >= budget or time.monotonic() >= deadline:
                break
            cur_x = lo + k * step
            cur_f = f(cur_x)
            counter["n"] += 1
            if _is_bad(cur_f):
                continue
            if abs(cur_f) <= _TOL:
                return (cur_x, cur_f)
            if cur_f * prev_f < 0.0:
                return (prev_x, cur_x)
            prev_x, prev_f = cur_x, cur_f

    return None


def find_root(f, x0):
    counter = {"n": 0}
    deadline = _START + _DEADLINE
    best = None
    best_af = float("inf")

    def upd(x, fx):
        nonlocal best, best_af
        if not _is_bad(x) and not _is_bad(fx) and abs(fx) < best_af:
            best_af = abs(fx)
            best = x

    def call(x):
        if counter["n"] >= _MAX_CALLS:
            return None
        counter["n"] += 1
        v = f(x)
        if v is None or not math.isfinite(v):
            return None
        return v

    # ---- Phase 1: bracket (or exact root) ----
    br = _find_bracket(f, x0, counter, _MAX_CALLS, deadline)
    if br is not None:
        if isinstance(br, tuple) and len(br) == 2 and isinstance(br[1], float):
            upd(br[0], br[1])
            return br[0]
        a, b = br
        fa = f(a)
        counter["n"] += 1
        fb = f(b)
        counter["n"] += 1
        if _is_bad(fa) or _is_bad(fb):
            return best if best is not None else float(x0)
        upd(a, fa)
        upd(b, fb)
        if fa * fb > 0.0:
            m = 0.5 * (a + b)
            fm = call(m)
            if fm is not None:
                upd(m, fm)
            return best if best is not None else m

        # ---- Phase 2: Newton-accelerated bisection ----
        x = 0.5 * (a + b)
        fx = call(x)
        if fx is None:
            fx = f(x)
            counter["n"] += 1
        if not _is_bad(fx):
            upd(x, fx)
        if fx is not None and abs(fx) <= _TOL:
            return x

        for _ in range(120):
            if counter["n"] >= _MAX_CALLS or time.monotonic() >= deadline:
                break
            if fx is None:
                break

            # Newton step (1 extra call for the derivative)
            h = 1e-5 * max(1.0, abs(x))
            fxp = call(x + h)
            if fxp is not None:
                d = (fxp - fx) / h
                if d != 0.0 and math.isfinite(d):
                    xn = x - fx / d
                    if math.isfinite(xn) and (a < xn < b):
                        fn = call(xn)
                        if fn is not None:
                            x, fx = xn, fn
                            upd(x, fx)
                            if abs(fx) <= _TOL:
                                return x
                            # keep the bracket as the sub-interval with a sign change
                            if fa * fn < 0.0:
                                b, fb = x, fx
                            else:
                                a, fa = x, fx
                            continue
                    elif math.isfinite(xn):
                        # outside bracket -> fall back to bisection below
                        pass

            # Bisection fallback (guaranteed progress)
            if fa * fx < 0.0:
                b, fb = x, fx
            else:
                a, fa = x, fx
            x = 0.5 * (a + b)
            fx = call(x)
            if fx is None:
                break
            upd(x, fx)
            if abs(fx) <= _TOL:
                return x

        return best if best is not None else x

    # ---- No bracket found: local Newton descent from x0 ----
    if not _is_bad(x0):
        x = float(x0)
        fx = call(x)
        if fx is not None:
            upd(x, fx)
        for _ in range(150):
            if counter["n"] >= _MAX_CALLS or time.monotonic() >= deadline:
                break
            if fx is None:
                break
            if abs(fx) <= _TOL:
                return x
            h = 1e-5 * max(1.0, abs(x))
            fxp = call(x + h)
            if fxp is None:
                break
            d = (fxp - fx) / h
            if d == 0.0 or not math.isfinite(d):
                break
            xn = x - fx / d
            if not math.isfinite(xn):
                break
            fn = call(xn)
            if fn is None:
                break
            x, fx = xn, fn
            upd(x, fx)
            if abs(fx) <= _TOL:
                return x
        return best if best is not None else x

    return best if best is not None else float(x0)
# EVOLVE-BLOCK-END
