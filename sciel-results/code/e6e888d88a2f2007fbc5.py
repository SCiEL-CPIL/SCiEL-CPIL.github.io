"""Compact dark-energy growth approximation: fit 6 coefficients to a local GR reference."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

_T0 = time.time()
_DEADLINE = _T0 + 80.0  # safety margin before the 90 s wall-clock kill
_CACHE = {}


def _om(a, Om0, wa):
    a = np.asarray(a, dtype=float)
    a3 = a ** 3
    de = (1.0 + wa) * a3 - wa
    return Om0 * a3 / (Om0 * a3 + (1.0 - Om0) * de)


def _gamma(c, a, Om0, wa):
    c0, c1, c2, c3, c4, c5 = c
    om = _om(a, Om0, wa)
    q = 1.0 - om
    u = 1.0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q ** 2 + c3 * u + c4 * wa * a + c5 * u * q


def _f(c, a, Om0, wa):
    om = _om(a, Om0, wa)
    g = _gamma(c, a, Om0, wa)
    return om ** g


def _D(c, a, Om0, wa):
    xs = np.linspace(np.log(a[0]), 0.0, 512)
    aa = np.exp(xs)
    fv = _f(c, aa, Om0, wa)
    # integral from ln(a0) to 0 of f dx = - integral over xs of f dlna
    I = -np.cumtrapz(fv, xs, initial=0.0)
    D = np.exp(-I)
    D /= D[-1]  # normalize D(1)=1
    idx = np.argmin(np.abs(xs - np.log(a)))
    return D[idx]


def _fit_one(Om0, wa, seed):
    a = np.linspace(0.08, 1.0, 220)

    def ode(x, y):
        aa = np.exp(x)
        om = _om(aa, Om0, wa)
        E = 1.5 * (1.0 + om)
        G = 1.5 * om
        return [y[1], -E * y[1] + G * y[0]]

    sol = solve_ivp(ode, [np.log(0.08), 0.0], [1.0, 1.0],
                    method="RK45", rtol=1e-9, atol=1e-12, dense_output=True)
    if not sol.success:
        return None
    ys = sol.sol(a)
    Dref = ys[0]
    Dref /= Dref[-1]
    fref = ys[1] / ys[0]

    def resid(c):
        om = _om(a, Om0, wa)
        g = _gamma(c, a, Om0, wa)
        f = om ** g
        D = _D(c, a, Om0, wa)
        e = np.concatenate([f / fref - 1.0, D / Dref - 1.0])
        pen = np.maximum(0.0, 0.1 - g) + np.maximum(0.0, g - 1.2)
        return np.concatenate([e, 20.0 * pen])

    try:
        r = least_squares(resid, seed, method="trf", max_nfev=800, xtol=1e-12, ftol=1e-12)
    except Exception:
        return None
    c = np.clip(r.x, -10.0, 10.0)
    # verify gamma constraint on a fine grid
    af = np.linspace(0.05, 1.0, 2000)
    gf = _gamma(c, af, Om0, wa)
    if gf.min() < 0.1 or gf.max() > 1.2:
        return None
    return c


def construct():
    global _CACHE
    if "c" in _CACHE:
        return _CACHE["c"]

    best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    models = [(0.3, 0.0), (0.3, -0.5), (0.27, -1.0), (0.3, 0.5), (0.3, -1.5)]
    for Om0, wa in models:
        if time.time() > _DEADLINE:
            break
        c = _fit_one(Om0, wa, best)
        if c is not None:
            best = c

    _CACHE["c"] = best
    return list(best)
