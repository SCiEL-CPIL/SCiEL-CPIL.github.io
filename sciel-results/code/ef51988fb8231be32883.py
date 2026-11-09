# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 8
_N2 = 64
_TUBE = 1.0 / _N


def _volume_fast(x, y):
    """Exact union volume of the 64 sheared boxes via interval merging."""
    a = x + np.arange(_N) * (_TUBE / _N)
    b = x + np.arange(_N) * (_TUBE / _N) + _TUBE
    c = y + np.arange(_N) * (_TUBE / _N)
    d = y + np.arange(_N) * (_TUBE / _N) + _TUBE
    xs = np.concatenate([a, b])
    xs = xs[np.isfinite(xs)]
    if xs.size < 2:
        return 0.0
    order = np.argsort(xs, kind="stable")
    xs = xs[order]
    lo = xs[0]
    hi = xs[-1]
    if hi <= lo + 1e-12:
        return 0.0
    nseg = 160
    t = np.linspace(0.0, 1.0, nseg + 1)
    segx = (hi - lo) / nseg
    acc = 0.0
    for k in range(nseg):
        cc = 0.5 * (t[k] + t[k + 1])
        # union of x-intervals at height cc
        xl = a + cc * np.arange(_N) * (_TUBE / _N)
        xr = xl + _TUBE
        orderx = np.argsort(xl, kind="stable")
        xl = xl[orderx]
        xr = xr[orderx]
        curL = xl[0]
        curR = xr[0]
        tot = 0.0
        for m in range(1, _N):
            L = xl[m]
            R = xr[m]
            if L > curR:
                tot += curR - curL
                curL = L
                curR = R
            else:
                if R > curR:
                    curR = R
        tot += curR - curL
        if tot <= 0.0:
            continue
        # union of y-intervals at height cc
        yl = c + cc * np.arange(_N) * (_TUBE / _N)
        yr = yl + _TUBE
        ordery = np.argsort(yl, kind="stable")
        yl = yl[ordery]
        yr = yr[ordery]
        curL = yl[0]
        curR = yr[0]
        toty = 0.0
        for m in range(1, _N):
            L = yl[m]
            R = yr[m]
            if L > curR:
                toty += curR - curL
                curL = L
                curR = R
            else:
                if R > curR:
                    curR = R
        toty += curR - curL
        if toty > 0.0:
            acc += segx * toty
    return acc


def _volume_gl(x, y):
    """Gauss-Legendre quadrature fallback (more accurate)."""
    xg, wg = np.polynomial.legendre.leggauss(24)
    t = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    a = x + np.arange(_N) * (_TUBE / _N)
    c = y + np.arange(_N) * (_TUBE / _N)
    tot = 0.0
    for k in range(t.size):
        cc = t[k]
        xl = a + cc * np.arange(_N) * (_TUBE / _N)
        xr = xl + _TUBE
        o = np.argsort(xl, kind="stable")
        xl = xl[o]
        xr = xr[o]
        curL, curR = xl[0], xr[0]
        sx = 0.0
        for m in range(1, _N):
            if xl[m] > curR:
                sx += curR - curL
                curL, curR = xl[m], xr[m]
            elif xr[m] > curR:
                curR = xr[m]
        sx += curR - curL
        if sx <= 0.0:
            continue
        yl = c + cc * np.arange(_N) * (_TUBE / _N)
        yr = yl + _TUBE
        o = np.argsort(yl, kind="stable")
        yl = yl[o]
        yr = yr[o]
        curL, curR = yl[0], yr[0]
        sy = 0.0
        for m in range(1, _N):
            if yl[m] > curR:
                sy += curR - curL
                curL, curR = yl[m], yr[m]
            elif yr[m] > curR:
                curR = yr[m]
        sy += curR - curL
        if sy > 0.0:
            tot += w[k] * sx * sy
    return tot


def _seed_positions():
    """Keich binary construction (the seed baseline)."""
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = keich[i_grid.flatten()]
    base_y = keich[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    deadline = start + 590.0  # safety margin before the 600s kill

    seed = _seed_positions()
    best = seed.copy()
    best_vol = _volume_fast(seed[:, 0], seed[:, 1])
    if not np.isfinite(best_vol) or best_vol <= 0.0:
        return seed

    def objective(p):
        return _volume_fast(p[:_N2], p[_N2:])

    def time_ok():
        return (time.time() - start) < deadline

    # ---- Phase 1: L-BFGS-B from the Keich seed ----
    try:
        res = minimize(
            objective,
            seed.ravel(),
            method="L-BFGS-B",
            options={
                "maxiter": 400,
                "maxfun": 400,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "disp": False,
            },
        )
        if res.x is not None:
            vol = _volume_fast(res.x[:_N2], res.x[_N2:])
            if np.isfinite(vol) and vol < best_vol:
                best = res.x.reshape(_N2, 2).copy()
                best_vol = vol
    except Exception:
        pass

    # ---- Phase 2: refine with Gauss-Legendre (higher fidelity) ----
    if time_ok():
        try:
            res = minimize(
                lambda p: _volume_gl(p[:_N2], p[_N2:]),
                best.ravel(),
                method="L-BFGS-B",
                options={
                    "maxiter": 300,
                    "maxfun": 300,
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                    "disp": False,
                },
            )
            if res.x is not None:
                vol = _volume_gl(res.x[:_N2], res.x[_N2:])
                if np.isfinite(vol) and vol < best_vol:
                    best = res.x.reshape(_N2, 2).copy()
                    best_vol = vol
        except Exception:
            pass

    # ---- Phase 3: short Nelder-Mead polish ----
    if time_ok():
        try:
            res = minimize(
                lambda p: _volume_gl(p[:_N2], p[_N2:]),
                best.ravel(),
                method="Nelder-Mead",
                options={
                    "maxiter": 400,
                    "maxfev": 400,
                    "xatol": 1e-6,
                    "fatol": 1e-9,
                    "disp": False,
                },
            )
            if res.x is not None:
                vol = _volume_gl(res.x[:_N2], res.x[_N2:])
                if np.isfinite(vol) and vol < best_vol:
                    best = res.x.reshape(_N2, 2).copy()
                    best_vol = vol
        except Exception:
            pass

    # ---- Phase 4: multi-restart random initializations if time remains ----
    rng = np.random.default_rng(20240501)
    for _ in range(6):
        if not time_ok():
            break
        p0 = seed.ravel().copy()
        p0 += rng.normal(0.0, 0.05, p0.size)
        try:
            res = minimize(
                lambda p: _volume_gl(p[:_N2], p[_N2:]),
                p0,
                method="L-BFGS-B",
                options={
                    "maxiter": 200,
                    "maxfun": 200,
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                    "disp": False,
                },
            )
            if res.x is not None:
                vol = _volume_gl(res.x[:_N2], res.x[_N2:])
                if np.isfinite(vol) and vol < best_vol:
                    best = res.x.reshape(_N2, 2).copy()
                    best_vol = vol
        except Exception:
            pass

    # Final validity guard: ensure finite and box area > 1e-9
    if not np.all(np.isfinite(best)):
        return seed
    X0 = best[:, 0].min()
    X1 = (best[:, 0] + np.arange(_N2) // _N * _TUBE).max()
    Y0 = best[:, 1].min()
    Y1 = (best[:, 1] + np.arange(_N2) % _N * _TUBE).max()
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return seed
    return best


# EVOLVE-BLOCK-END
