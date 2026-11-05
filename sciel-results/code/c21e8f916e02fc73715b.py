"""Cosmology: compact dark-energy growth-index fit.
SCiEL evolves the EVOLVE-BLOCK.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.integrate import solve_ivp

_T0 = time.time()
_HARD = 86.0
_SOFT = 82.0


def _deadline(kind="soft"):
    lim = _SOFT if kind == "soft" else _HARD
    return (time.time() - _T0) >= lim


def _H2(a, Om0, w0, wa):
    Ed = Om0 * a ** (-3) + (1.0 - Om0) * a ** (3.0 * (1.0 + w0 + wa * (1.0 - a)))
    return 1.0 / Ed


def _Fprime(a, Om0, w0, wa):
    H2 = _H2(a, Om0, w0, wa)
    Ed = 1.0 / H2
    w = w0 + wa * (1.0 - a)
    Edp = -3.0 * Om0 * a ** (-4) + 3.0 * (1.0 - Om0) * a ** (3.0 * (1.0 + w)) * ((1.0 + w) - wa)
    dlnH = 0.5 * Edp / Ed
    return (3.0 / (2.0 * a) - dlnH) * 0.5 - 1.5 * H2


def _integrate_F(a_grid, Om0, w0, wa):
    F0 = 0.5 * a_grid[0] ** 2
    dF0 = 1.5 * a_grid[0] ** 3
    sol = solve_ivp(
        lambda x, y: [y[1], _Fprime(math.exp(x), Om0, w0, wa) * y[1] - 1.5 * _H2(math.exp(x), Om0, w0, wa) * y[0]],
        [math.log(a_grid[0]), math.log(a_grid[-1])],
        [F0, dF0], method="RK45", rtol=1e-10, atol=1e-13,
    )
    return sol.y[0]


def _cosmo_set():
    grid = np.exp(np.linspace(np.log(0.05), 0.0, 90))
    out = []
    for Om0 in (0.20, 0.26, 0.32):
        for w0 in (-1.30, -1.00, -0.70):
            for wa in (-0.50, 0.0, 0.50):
                out.append((grid, Om0, w0, wa))
    return out


def _fit_data():
    cache = {}
    for grid, Om0, w0, wa in _cosmo_set():
        if _deadline():
            break
        F = _integrate_F(grid, Om0, w0, wa)
        Fn = F / F[-1]
        Fp = np.gradient(Fn, grid)
        Om = Om0 * grid ** (-3)
        f_ref = Fp / Om
        f_ref = np.clip(f_ref, 1e-4, 1e3)
        Dn = np.clip(Fn, 1e-6, None)
        cache[(Om0, w0, wa)] = (grid, Om, f_ref, Dn)
    return cache


def _objective(c, cache, S):
    c0, c1, c2, c3, c4, c5 = c
    logsum = -1e300
    hardmax = 0.0
    gmin = 1e9
    gmax = -1e9
    for key, (a, Om, f_ref, Dn) in cache.items():
        q = 1.0 - Om
        u = 1.0 + key[1] + key[2] * (1.0 - a)
        g = c0 + c1 * q + c2 * q ** 2 + c3 * u + c4 * key[2] * a + c5 * u * q
        gmin = min(gmin, g.min())
        gmax = max(gmax, g.max())
        f = Om ** g
        r1 = np.abs(f - f_ref) / f_ref
        gD = np.cumsum(f * np.diff(np.log(a)))
        D = np.exp(-gD)
        r2 = np.abs(D - Dn) / Dn
        r = np.maximum(r1, r2)
        r = np.maximum(r, 1e-8)
        m = r.max()
        hardmax = max(hardmax, m)
        Smax = (1.0 / S) * np.log(np.sum(np.exp(S * r)))
        logsum = math.logaddexp(logsum, Smax)
    smooth = (1.0 / S) * math.logaddexp(0.0, logsum)
    penalty = 0.0
    if gmin < 0.1:
        penalty += 50.0 * (0.1 - gmin) ** 2
    if gmax > 1.2:
        penalty += 50.0 * (gmax - 1.2) ** 2
    val = 100.0 * smooth + penalty
    return val, 100.0 * hardmax + penalty


def _nelder_mead(f, x0, bounds, maxfev, ftol, xatol):
    lo = np.array(bounds[0], float)
    hi = np.array(bounds[1], float)
    x0 = np.clip(x0, lo, hi)
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] = p[i] + 0.25 + 0.1 * i
        simplex.append(np.clip(p, lo, hi))
    fs = [f(v) for v in simplex]
    for _ in range(maxfev):
        order = np.argsort(fs)
        simplex = [simplex[i] for i in order]
        fs = [fs[i] for i in order]
        if (fs[-1] - fs[0]) < ftol * (abs(fs[0]) + 1e-12):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = np.clip(centroid + 1.0 * (centroid - simplex[-1]), lo, hi)
        fr = f(xr)
        if fr < fs[0]:
            xe = np.clip(centroid + 2.0 * (xr - centroid), lo, hi)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fs[-1] = xe, fe
            else:
                simplex[-1], fs[-1] = xr, fr
        elif fr < fs[-2]:
            simplex[-1], fs[-1] = xr, fr
        else:
            xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), lo, hi)
            fc = f(xc)
            if fc < fs[-1]:
                simplex[-1], fs[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), lo, hi)
                    fs[i] = f(simplex[i])
    best = int(np.argmin(fs))
    return simplex[best], fs[best]


def construct():
    cache = _fit_data()
    if not cache:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = ([-10.0] * 6, [10.0] * 6)
    S = 8.0
    Ss = [8.0, 16.0, 32.0, 64.0]
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_v, best_h = _objective(best_c, cache, Ss[-1])

    def fobj(v):
        return _objective(v, cache, Ss[-1])[0]

    # Coarse grid to seed multistart.
    grid_c0 = np.linspace(0.25, 0.90, 8)
    grid_c1 = np.linspace(-0.8, 0.8, 5)
    grid_c3 = np.linspace(-0.5, 0.5, 5)
    grid_c4 = np.linspace(-0.5, 0.5, 5)
    nfe = 0
    maxfev = 220
    for c0 in grid_c0:
        for c1 in grid_c1:
            for c3 in grid_c3:
                for c4 in grid_c4:
                    if _deadline("soft"):
                        break
                    if nfe >= maxfev:
                        break
                    x0 = np.array([c0, c1, 0.0, c3, c4, 0.0])
                    xc, fc = _nelder_mead(fobj, x0, bounds, maxfev=maxfev, ftol=1e-9, xatol=1e-10)
                    nfe += 1
                    fh = _objective(xc, cache, Ss[-1])[1]
                    if fc < best_v:
                        best_c, best_v, best_h = xc, fc, fh
                if _deadline("soft") or nfe >= maxfev:
                    break
            if _deadline("soft") or nfe >= maxfev:
                break
        if _deadline("soft") or nfe >= maxfev:
            break

    # Final high-precision polish with tighter tolerances.
    if not _deadline("soft"):
        xc, fc = _nelder_mead(fobj, best_c, bounds, maxfev=500, ftol=1e-12, xatol=1e-12)
        fh = _objective(xc, cache, Ss[-1])[1]
        if fc < best_v:
            best_c, best_v, best_h = xc, fc, fh

    out = np.clip(best_c, -10.0, 10.0).tolist()
    return out
# EVOLVE-BLOCK-END
