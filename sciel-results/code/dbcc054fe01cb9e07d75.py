import time
import math
import numpy as np
import scipy.optimize

_DEADLINE = 150.0
_MARGIN = 25.0          # safety margin before wall-clock kill
_EVAL_BUDGET = 490      # hard cap on f() evaluations (task allows 500)
_ABS_TOL = 1e-8


class _Budget:
    __slots__ = ("count", "start", "ok")

    def __init__(self):
        self.count = 0
        self.start = time.time()
        self.ok = True

    def eval(self, f, x):
        if not self.ok:
            return 0.0
        self.count += 1
        if self.count > _EVAL_BUDGET or (time.time() - self.start) > (_DEADLINE - _MARGIN):
            self.ok = False
            return 0.0
        try:
            v = f(float(x))
        except Exception:
            self.ok = False
            return 0.0
        if not math.isfinite(v):
            return float("inf")
        return v


def _bracket_sign(f, x0, fx0, B):
    """Find [a,b] with a sign change of f. Returns (a,fa,b,fb) or None."""
    a = float(x0)
    fa = fx0
    h = 1.0
    for _ in range(40):
        if not B.ok:
            return None
        xr = a + h
        fr = B.eval(f, xr)
        if fr != float("inf") and fa * fr <= 0:
            return a, fa, xr, fr
        xl = a - h
        fl = B.eval(f, xl)
        if fl != float("inf") and fa * fl <= 0:
            return xl, fl, a, fa
        h *= 2.0
        if h > 1e100:
            return None
    return None


def _bracket_min(f, x0, fx0, B):
    """Find [a,b] where |f| at x0 is a strict local min. Returns (a,b) or None."""
    gx0 = fx0 * fx0
    h = 1.0
    for _ in range(40):
        if not B.ok:
            return None
        xl = x0 - h
        xr = x0 + h
        fl = B.eval(f, xl)
        fr = B.eval(f, xr)
        gl = fl * fl
        gr = fr * fr
        if gx0 < gl and gx0 < gr:
            return xl, xr
        h *= 2.0
        if h > 1e100:
            return None
    return None


def _minimize_scalar(f, a, b, B):
    """Golden-section minimization of g(x)=f(x)^2 on [a,b]. Returns (x, g)."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = B.eval(f, c)
    fd = B.eval(f, d)
    gc = fc * fc
    gd = fd * fd
    for _ in range(200):
        if not B.ok:
            break
        if gc < gd:
            b = d
            d = c
            gd = gc
            c = b - gr * (b - a)
            gc = B.eval(f, c)
            gc = gc * gc
        else:
            a = c
            c = d
            gc = gd
            d = a + gr * (b - a)
            gd = B.eval(f, d)
            gd = gd * gd
        if abs(b - a) < 1e-13 * max(1.0, abs(c)):
            break
    # return the better of the two interior points
    if gc <= gd:
        return c, gc
    return d, gd


def _safeguarded_secant(f, x0, fx0, B):
    """Secant/Newton with bisection-on-minimum safeguard. Returns (x, fx)."""
    x = float(x0)
    fx = fx0
    # seed a second point for the secant
    h = 1e-4 * max(1.0, abs(x))
    xp = B.eval(f, x + h)
    if xp == float("inf"):
        xp = B.eval(f, x - h)
        xprev = x - h
    else:
        xprev = x + h
    xprev_f = xp
    best_x = x
    best_f = fx
    step = 0.0
    for _ in range(200):
        if not B.ok:
            break
        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        if abs(fx) <= _ABS_TOL:
            return x, fx
        # secant direction
        if xprev_f != fx:
            slope = (fx - xprev_f) / (x - xprev)
        else:
            slope = 1e-12
        if slope == 0.0:
            break
        newton = x - fx / slope
        # safeguard: keep newton within [xprev, x] (bisection if outside)
        lo, hi = (xprev, x) if xprev < x else (x, xprev)
        mid = 0.5 * (lo + hi)
        if newton < lo or newton > hi:
            newton = mid
        fnew = B.eval(f, newton)
        if fnew == float("inf"):
            # step was bad; shrink toward midpoint
            newton = mid
            fnew = B.eval(f, newton)
            if fnew == float("inf"):
                break
        # advance the secant pair
        xprev = x
        xprev_f = fx
        x = newton
        fx = fnew
        # convergence on x
        if abs(step) < 1e-13 * max(1.0, abs(x)):
            break
        step = x - xprev
    if abs(fx) < abs(best_f):
        best_f = fx
        best_x = x
    return best_x, best_f


def _newton_with_central(f, x0, B):
    """Newton with central-difference derivative and damping. Returns (x, fx)."""
    x = float(x0)
    fx = B.eval(f, x)
    if fx == float("inf"):
        return x, fx
    best_x = x
    best_f = fx
    for _ in range(100):
        if not B.ok:
            break
        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        if abs(fx) <= _ABS_TOL:
            return x, fx
        h = 1e-8 * max(1.0, abs(x))
        fpx = B.eval(f, x + h)
        fxm = B.eval(f, x - h)
        if fpx != float("inf") and fxm != float("inf"):
            d = (fpx - fxm) / (2.0 * h)
        elif fpx != float("inf"):
            d = (fpx - fx) / h
        elif fxm != float("inf"):
            d = (fx - fxm) / h
        else:
            break
        if d == 0.0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        # damping
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
        if not math.isfinite(x_new):
            break
        fx_new = B.eval(f, x_new)
        if fx_new == float("inf"):
            break
        if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
            x = x_new
            fx = fx_new
            break
        x = x_new
        fx = fx_new
    if abs(fx) < abs(best_f):
        best_f = fx
        best_x = x
    return best_x, best_f


def find_root(f, x0):
    B = _Budget()
    x0 = float(x0)
    fx0 = B.eval(f, x0)
    if not B.ok:
        return x0
    if abs(fx0) <= _ABS_TOL:
        return x0

    best_x, best_f = x0, fx0

    # 1) Quick sign-change bracket -> brentq (most reliable, fewest evals)
    br = _bracket_sign(f, x0, fx0, B)
    if br is not None:
        a, fa, b, fb = br
        try:
            r = scipy.optimize.brentq(
                lambda z: B.eval(f, z), a, b,
                xtol=1e-14, rtol=1e-14, maxiter=100
            )
            fr = B.eval(f, r)
            if abs(fr) < abs(best_f):
                best_x, best_f = r, fr
            if abs(fr) <= _ABS_TOL:
                return r
        except Exception:
            pass

    # 2) Multi-start safeguarded secant from several perturbed points
    #    For ill-conditioned problems, the basin of attraction is narrow,
    #    so trying multiple starting points with one robust method is
    #    more effective than trying many different methods from one point.
    scale = max(1.0, abs(best_x))
    perturbations = [0.0, 1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 10.0, -10.0]
    
    for pert in perturbations:
        if not B.ok:
            break
        if (time.time() - B.start) > (_DEADLINE - _MARGIN):
            break
        # Check budget: need at least ~50 evals remaining for a meaningful run
        if B.count > _EVAL_BUDGET - 50:
            break
        
        x_start = best_x + pert * scale
        try:
            fx_start = B.eval(f, x_start)
            if fx_start == float("inf"):
                continue
            if abs(fx_start) < abs(best_f):
                best_x, best_f = x_start, fx_start
            if abs(fx_start) <= _ABS_TOL:
                return x_start
            
            xr, fxr = _safeguarded_secant(f, x_start, fx_start, B)
            if abs(fxr) < abs(best_f):
                best_x, best_f = xr, fxr
            if abs(best_f) <= _ABS_TOL:
                return best_x
        except Exception:
            pass

    # 3) Final polish: try Newton with central difference from the best point
    if B.ok and B.count < _EVAL_BUDGET - 20:
        try:
            xr, fxr = _newton_with_central(f, best_x, B)
            if abs(fxr) < abs(best_f):
                best_x, best_f = xr, fxr
            if abs(best_f) <= _ABS_TOL:
                return best_x
        except Exception:
            pass

    return best_x
