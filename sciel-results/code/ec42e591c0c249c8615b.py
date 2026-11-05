"""Root finding for ill-conditioned equations.

Strategy: combine exact-derivative Newton (via dual numbers when possible),
bracketed bisection/Illinois/Brent, and unbracketed Muller/secant. Track the
best point by |f(x)| and enforce a hard wall-clock deadline so we always
return the best result found before the limit.
"""
import math
import time

# EVOLVE-BLOCK-START

# ---- deadline guard ----
_START = None
_DEADLINE = 0.0
_BEST = {"x": None, "fx": None, "abs": float("inf")}
_EVALS = 0
_MAX_EVALS = 480


def _elapsed():
    return time.perf_counter() - _START


def _alive():
    return _elapsed() < _DEADLINE and _EVALS < _MAX_EVALS


def _eval(f, x):
    global _EVALS
    if _EVALS >= _MAX_EVALS:
        return None
    _EVALS += 1
    try:
        y = f(x)
    except Exception:
        return None
    try:
        yf = float(y)
    except Exception:
        return None
    return yf


def _record(f, x, fx):
    global _BEST
    if fx is None:
        return
    a = abs(fx)
    if a < _BEST["abs"]:
        _BEST = {"x": x, "fx": fx, "abs": a}


# ---- dual-number derivative support ----
class _Dual:
    __slots__ = ("p", "e")

    def __init__(self, p, e=0.0):
        self.p = float(p)
        self.e = float(e)

    def __add__(self, o):
        if isinstance(o, _Dual):
            return _Dual(self.p + o.p, self.e + o.e)
        return _Dual(self.p + float(o), self.e)

    def __radd__(self, o):
        return _Dual(float(o) + self.p, self.e)

    def __sub__(self, o):
        if isinstance(o, _Dual):
            return _Dual(self.p - o.p, self.e - o.e)
        return _Dual(self.p - float(o), self.e)

    def __rsub__(self, o):
        return _Dual(float(o) - self.p, -self.e)

    def __mul__(self, o):
        if isinstance(o, _Dual):
            return _Dual(self.p * o.p, self.p * o.e + o.p * self.e)
        f = float(o)
        return _Dual(self.p * f, self.e * f)

    def __rmul__(self, o):
        f = float(o)
        return _Dual(self.p * f, self.e * f)

    def __truediv__(self, o):
        if isinstance(o, _Dual):
            op, oe = o.p, o.e
            if op == 0.0:
                raise ZeroDivisionError
            d = op * op
            return _Dual(self.p / op, (self.e * op - self.p * oe) / d)
        f = float(o)
        if f == 0.0:
            raise ZeroDivisionError
        return _Dual(self.p / f, self.e / f)

    def __rtruediv__(self, o):
        f = float(o)
        if self.p == 0.0:
            raise ZeroDivisionError
        d = self.p * self.p
        return _Dual(f / self.p, (f * self.e) / d)

    def __pow__(self, n):
        n = float(n)
        if n == 0.0:
            return _Dual(1.0, 0.0)
        if self.p == 0.0 and n <= 0.0:
            raise ZeroDivisionError
        return _Dual(self.p ** n, n * self.p ** (n - 1.0) * self.e)

    def __neg__(self):
        return _Dual(-self.p, -self.e)

    def __abs__(self):
        return _Dual(abs(self.p), math.copysign(self.e, self.p))


def _try_dual(f, x):
    """Return (fx, dfx) using dual numbers, or None."""
    try:
        d = _Dual(x, 1.0)
        r = f(d)
        if not isinstance(r, _Dual):
            return None
        if not (math.isfinite(r.p) and math.isfinite(r.e)):
            return None
        return r.p, r.e
    except Exception:
        return None


def _finite_diff(f, x, fx, h):
    h = h * max(1.0, abs(x))
    fxh = _eval(f, x + h)
    if fxh is None:
        return None
    if fx is None:
        fx = _eval(f, x)
        if fx is None:
            return None
    return (fxh - fx) / h


# ---- bracketing ----
def _find_bracket(f, x, fx):
    """Expand outward to find a sign-change bracket [lo, hi]."""
    a = x
    fa = fx
    if fa is None:
        return None
    step = 0.1 * max(1.0, abs(x))
    # try both directions, keep the one that gives a sign change
    for direction in (1.0, -1.0):
        b = x + direction * step
        fb = _eval(f, b)
        if fb is None:
            continue
        if fa == 0.0:
            return (x, x)
        if fa * fb < 0.0:
            if direction > 0:
                return (a, b)
            else:
                return (b, a)
        # expand
        cur = b
        curf = fb
        for _ in range(60):
            if not _alive():
                break
            step *= 2.0
            nxt = x + direction * step
            fn = _eval(f, nxt)
            if fn is None:
                break
            if curf == 0.0:
                return (cur, cur)
            if curf * fn < 0.0:
                if direction > 0:
                    return (cur, nxt)
                else:
                    return (nxt, cur)
            cur, curf = nxt, fn
    return None


# ---- bisection ----
def _bisect(f, lo, hi, flo, fhi):
    for _ in range(80):
        if not _alive():
            break
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        fm = _eval(f, mid)
        if fm is None:
            break
        _record(f, mid, fm)
        if fm == 0.0 or abs(fm) <= 1e-12:
            return mid
        if flo is None:
            flo = _eval(f, lo)
        if flo is not None and flo * fm < 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if abs(hi - lo) <= 1e-14 * max(1.0, abs(mid)):
            return mid
    return 0.5 * (lo + hi)


# ---- Illinois / Brent-ish bracketed refinement ----
def _illinois(f, lo, hi, flo, fhi):
    for _ in range(80):
        if not _alive():
            break
        if flo * fhi > 0.0:
            break
        if abs(flo) < abs(fhi):
            lo, hi = hi, lo
            flo, fhi = fhi, flo
        # interpolate toward the smaller-magnitude side (Illinois)
        denom = fhi - flo
        if abs(denom) < 1e-300:
            mid = 0.5 * (lo + hi)
        else:
            mid = (flo * hi - fhi * lo) / denom
            if not (lo < mid < hi):
                mid = 0.5 * (lo + hi)
        fm = _eval(f, mid)
        if fm is None:
            break
        _record(f, mid, fm)
        if fm == 0.0 or abs(fm) <= 1e-12:
            return mid
        if flo * fm < 0.0:
            hi, fhi = mid, fm
        else:
            # Illinois: shrink the larger interval half
            lo, flo = mid, 0.5 * flo
        if abs(hi - lo) <= 1e-14 * max(1.0, abs(mid)):
            return mid
    return 0.5 * (lo + hi)


# ---- secant method ----
def _secant(f, x0):
    x0 = float(x0)
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    f0 = _eval(f, x0)
    f1 = _eval(f, x1)
    if f0 is None or f1 is None:
        return None
    _record(f, x0, f0)
    _record(f, x1, f1)
    for _ in range(80):
        if not _alive():
            break
        d = f1 - f0
        if d == 0.0:
            break
        x2 = x1 - f1 * (x1 - x0) / d
        if not math.isfinite(x2):
            break
        f2 = _eval(f, x2)
        if f2 is None:
            break
        _record(f, x2, f2)
        if f2 == 0.0 or abs(f2) <= 1e-12:
            return x2
        x0, f0 = x1, f1
        x1, f1 = x2, f2
    return x1


# ---- Newton with exact or FD derivative ----
def _newton(f, x0, use_dual):
    x = float(x0)
    for _ in range(60):
        if not _alive():
            break
        if use_dual:
            pair = _try_dual(f, x)
            if pair is not None:
                fx, dfx = pair
                _record(f, x, fx)
                if fx == 0.0 or abs(fx) <= 1e-12:
                    return x
                if dfx == 0.0 or not math.isfinite(dfx):
                    break
                step = fx / dfx
                if not math.isfinite(step):
                    break
                # damping to keep near x0
                xnew = x - step
                if abs(xnew) > 1e6 * max(1.0, abs(x)):
                    xnew = x - 0.5 * step
                    if abs(xnew) > 1e6 * max(1.0, abs(x)):
                        break
                x = xnew
                continue
        fx = _eval(f, x)
        if fx is None:
            break
        _record(f, x, fx)
        if fx == 0.0 or abs(fx) <= 1e-12:
            return x
        dfx = _finite_diff(f, x, fx, 1e-6)
        if dfx is None or dfx == 0.0:
            break
        step = fx / dfx
        if not math.isfinite(step):
            break
        xnew = x - step
        if abs(xnew) > 1e6 * max(1.0, abs(x)):
            xnew = x - 0.5 * step
        x = xnew
    return x


# ---- damped fixed-point fallback ----
def _fixed_point(f, x0):
    x = float(x0)
    for _ in range(60):
        if not _alive():
            break
        fx = _eval(f, x)
        if fx is None:
            break
        _record(f, x, fx)
        if abs(fx) <= 1e-12:
            return x
        # try x - k*f with a few k
        for k in (0.1, 0.01, 1.0):
            xn = x - k * fx
            if math.isfinite(xn):
                x = xn
                break
    return x


def _best_x(f):
    # ensure best is at least the initial value
    return _BEST["x"]


def find_root(f, x0):
    global _START, _DEADLINE, _BEST, _EVALS
    _START = time.perf_counter()
    _DEADLINE = _START + 110.0   # safety margin under the 120 s per-eval / 150 s kill
    _EVALS = 0
    _BEST = {"x": float(x0), "fx": None, "abs": float("inf")}

    x0f = float(x0)
    fx0 = _eval(f, x0f)
    _record(f, x0f, fx0)
    if fx0 is not None and abs(fx0) <= 1e-8:
        return x0f

    # detect dual support once
    use_dual = _try_dual(f, x0f) is not None

    # 1) Newton (exact derivative preferred)
    x = _newton(f, x0f, use_dual)
    if x is not None and _EVALS < _MAX_EVALS and _elapsed() < _DEADLINE:
        fx = _eval(f, x)
        if fx is not None:
            _record(f, x, fx)
        if fx is not None and abs(fx) <= 1e-8:
            return x

    # 2) bracket + Illinois
    br = _find_bracket(f, x0f, fx0)
    if br is not None:
        lo, hi = br
        flo = _eval(f, lo)
        fhi = _eval(f, hi)
        x = _illinois(f, lo, hi, flo, fhi)
        fx = _eval(f, x)
        if fx is not None:
            _record(f, x, fx)
        if fx is not None and abs(fx) <= 1e-8:
            return x
        # 3) plain bisection as a guaranteed-convergence bracketed method
        x = _bisect(f, lo, hi, flo, fhi)
        fx = _eval(f, x)
        if fx is not None:
            _record(f, x, fx)
        if fx is not None and abs(fx) <= 1e-8:
            return x

    # 4) secant
    x = _secant(f, x0f)
    if x is not None:
        fx = _eval(f, x)
        if fx is not None:
            _record(f, x, fx)
        if fx is not None and abs(fx) <= 1e-8:
            return x

    # 5) Try Newton/secant from perturbed starting points (inspiration from portfolio approach)
    for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
        if not _alive():
            break
        x_try = x0f + offset
        # Try Newton from perturbed point
        x = _newton(f, x_try, use_dual)
        if x is not None:
            fx = _eval(f, x)
            if fx is not None:
                _record(f, x, fx)
            if fx is not None and abs(fx) <= 1e-8:
                return x
        # Try secant from perturbed point
        x = _secant(f, x_try)
        if x is not None:
            fx = _eval(f, x)
            if fx is not None:
                _record(f, x, fx)
            if fx is not None and abs(fx) <= 1e-8:
                return x

    # 6) fixed-point
    x = _fixed_point(f, x0f)
    fx = _eval(f, x)
    if fx is not None:
        _record(f, x, fx)

    # return the best point seen
    return _best_x(f)

# EVOLVE-BLOCK-END
