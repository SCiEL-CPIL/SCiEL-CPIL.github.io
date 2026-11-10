# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 8
_NN = _N * _N
_EPS = 1e-9
_TIME_LIMIT = 620.0  # safety margin under the 630s kill


def _critical_c(i, j, x0, y0):
    """All c in [0,1] where tube (i,j) enters/leaves the reference unit cell
    [0,1]x[0,1], plus the endpoints 0 and 1."""
    lox, hix = x0, x0 + 1.0 / _N
    loy, hiy = y0, y0 + 1.0 / _N
    # x interval: [lox + c*i/N, hix + c*i/N] overlaps [0,1] iff
    #   lox + c*i/N < 1  and  hix + c*i/N > 0
    cs = [0.0, 1.0]
    if i > 0:
        cs.append((1.0 - lox) / (i / _N))
        cs.append((-hix) / (i / _N))
    if j > 0:
        cs.append((1.0 - loy) / (j / _N))
        cs.append((-hiy) / (j / _N))
    cs = [c for c in cs if _EPS < c < 1.0 - _EPS]
    cs.sort()
    return np.array([0.0] + cs + [1.0])


def _cell_volume(i, j, x0, y0):
    """Exact integral over c in [0,1] of (N_tubes(c) - 1) for the reference cell
    [0,1]x[0,1] (a unit x-unit x-unit cell)."""
    lox, hix = x0, x0 + 1.0 / _N
    loy, hiy = y0, y0 + 1.0 / _N
    crit = _critical_c(i, j, lox, loy)
    if crit.size <= 1:
        return 0.0
    mid = 0.5 * (crit[:-1] + crit[1:])
    widths = np.diff(crit)
    if widths.size == 0:
        return 0.0
    # x overlap at height c: [lox + c*i/N, hix + c*i/N] vs [0,1]
    x_lo = lox + mid * i / _N
    x_hi = hix + mid * i / _N
    y_lo = loy + mid * j / _N
    y_hi = hiy + mid * j / _N
    ox = np.minimum(1.0, x_hi) - np.minimum(x_lo, 1.0)
    oy = np.minimum(1.0, y_hi) - np.minimum(y_lo, 1.0)
    inx = np.clip(ox, 0.0, None)
    iny = np.clip(oy, 0.0, None)
    n = np.clip(inx * iny, 0.0, 1.0)
    return float(np.sum(widths * np.clip(n - 1.0, 0.0, None)))


def _grid_bounds(x, y):
    """Bounding box in x,y for the union of all 64 tubes (full height)."""
    X0 = np.min(x)
    X1 = np.max(x + np.arange(_N)[None, :] / _N)  # per i
    X1 = np.max(x + (np.arange(_N) / _N))
    Y0 = np.min(y)
    Y1 = np.max(y + (np.arange(_N) / _N))
    return X0, X1, Y0, Y1


def _exact_volume(x, y):
    """Exact union volume of the 64 sheared tubes via cell-wise critical-c
    integration. x, y: arrays of shape (N,) (per-index base positions)."""
    X0, X1, Y0, Y1 = _grid_bounds(x, y)
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return float('inf')
    nx = int(np.ceil(X1 - X0))
    ny = int(np.ceil(Y1 - Y0))
    if nx < 1:
        nx = 1
    if ny < 1:
        ny = 1
    if nx * ny > 400000:  # guard against pathological blow-up
        return float('inf')
    total = 0.0
    for ix in range(nx):
        cx0 = X0 + ix
        for iy in range(ny):
            cy0 = Y0 + iy
            for i in range(_N):
                for j in range(_N):
                    x0 = x[i] - cx0
                    y0 = y[j] - cy0
                    # quick reject: tube (i,j) never overlaps this cell?
                    # x overlap over c in [0,1]: union of intervals
                    if i == 0:
                        xlo, xhi = x0, x0 + 1.0 / _N
                    else:
                        xlo = x0
                        xhi = x0 + 1.0 / _N + i / _N
                        xlo2 = x0 + i / _N
                        # overall x extent = [min(x0, x0+i/N), max(x0+1/N, x0+i/N+1/N)]
                        xlo = min(x0, x0 + i / _N)
                        xhi = max(x0 + 1.0 / _N, x0 + i / _N + 1.0 / _N)
                    if j == 0:
                        ylo, yhi = y0, y0 + 1.0 / _N
                    else:
                        ylo = min(y0, y0 + j / _N)
                        yhi = max(y0 + 1.0 / _N, y0 + j / _N + 1.0 / _N)
                    if xhi <= 0.0 or yhi <= 0.0 or xlo >= 1.0 or ylo >= 1.0:
                        continue
                    total += _cell_volume(i, j, x0, y0)
    return total


def _mc_volume(positions, n=200000, seed=12345):
    """Monte Carlo estimate of union volume (for sanity checking)."""
    N = _N
    i = np.arange(_NN) // N
    j = np.arange(_NN) % N
    xk = positions[:, 0]
    yk = positions[:, 1]
    X0 = np.min(xk)
    X1 = np.max(xk + (i + 1) / N)
    Y0 = np.min(yk)
    Y1 = np.max(yk + (j + 1) / N)
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return float('inf')
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n)
    b = rng.uniform(Y0, Y1, n)
    c = rng.uniform(0.0, 1.0, n)
    # inside any tube?
    # x in [xk + c*i/N, xk + c*i/N + 1/N]  =>  a - c*i/N in [xk, xk+1/N]
    # vectorize over tubes via broadcasting (n x NN) - do in chunks
    inside = np.zeros(n, dtype=bool)
    chunk = 20000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        aa = a[s:e][:, None]
        bb = b[s:e][:, None]
        cc = c[s:e][:, None]
        xx = xk[None, :] + cc * i[None, :] / N
        yy = yk[None, :] + cc * j[None, :] / N
        inx = (aa >= xx) & (aa <= xx + 1.0 / N)
        iny = (bb >= yy) & (bb <= yy + 1.0 / N)
        inside[s:e] = np.any(inx & iny, axis=1)
    frac = np.mean(inside)
    return float(frac * (X1 - X0) * (Y1 - Y0))


def _keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _keich_positions():
    keich = _keich(_N)
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    return np.stack([keich[i_grid.flatten()], keich[j_grid.flatten()]], axis=1)


def _product_positions(x, y):
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    return np.stack([x[i_grid.flatten()], y[j_grid.flatten()]], axis=1)


def _full_positions(p):
    """p: (128,) -> (64,2) positions."""
    x = p[:_NN]
    y = p[_NN:]
    return np.stack([x, y], axis=1)


def _objective_product(v):
    x = v[:_N]
    y = v[_N:]
    return _exact_volume(x, y)


def _objective_full(v):
    positions = _full_positions(v)
    x = positions[:, 0]
    y = positions[:, 1]
    return _exact_volume(x, y)


def _keich_flat_product():
    keich = _keich(_N)
    return np.concatenate([keich, keich])


def _keich_flat_full():
    pos = _keich_positions()
    return pos.flatten()


def kakeya_needle_3d8() -> np.ndarray:
    t0 = time.time()
    keich = _keich(_N)
    best_pos = _keich_positions()
    best_val = _exact_volume(keich, keich)
    if not np.isfinite(best_val):
        best_val = _mc_volume(best_pos)

    # ---- Phase A: product-form Powell (16 params), multiple restarts ----
    def restart_product(center):
        v0 = np.array(center, dtype=float)
        try:
            res = minimize(_objective_product, v0, method="Powell",
                           options={"maxiter": 60, "xtol": 1e-9, "ftol": 1e-12,
                                    "disp": False})
            return res.x, res.fun
        except Exception:
            return v0, _objective_product(v0)

    restarts = [_keich_flat_product()]
    # a few deterministic jittered restarts
    rng = np.random.default_rng(7)
    for _ in range(4):
        restarts.append(_keich_flat_product() + rng.uniform(-0.25, 0.25, 16))

    for r in restarts:
        if time.time() - t0 > _TIME_LIMIT - 120:
            break
        xv, val = restart_product(r)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_pos = _product_positions(xv[:_N], xv[_N:])

    # ---- Phase B: full 128-param polish from best product solution ----
    if time.time() - t0 < _TIME_LIMIT - 60:
        p0 = best_pos.flatten().copy()
        try:
            res = minimize(_objective_full, p0, method="Powell",
                           options={"maxiter": 80, "xtol": 1e-9, "ftol": 1e-12,
                                    "disp": False})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_pos = _full_positions(res.x)
        except Exception:
            pass

    # ---- Final validity check via the task's Monte Carlo definition ----
    if time.time() - t0 < _TIME_LIMIT - 20:
        mc = _mc_volume(best_pos, n=200000, seed=12345)
        if not np.isfinite(best_val) or best_val <= 0 or not np.isfinite(mc):
            # fall back to keich
            return _keich_positions()

    return best_pos


# EVOLVE-BLOCK-END
