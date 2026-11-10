# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.optimize import minimize

_N = 8
_S = 1.0 / 8.0
_NN = 64
# deadline guard: the evaluator kills the process at ~630s; finish well before.
_DEADLINE = time.time() + 600.0
_MARGIN = 30.0


def _union_len(base, idx):
    """Exact 1D union length at height c for one family (x or y).

    base: shape (N,), base coordinate per grid index.
    idx:  shape (N*N,), the i (or j) index for each tube.
    Returns length of union of intervals [base[idx_k] + c*idx_k/N, +1/N]
    for c of shape (Mc,).
    """
    s = c * idx * _S
    lo = base[idx] + s
    hi = lo + _S
    order = np.argsort(lo, axis=1)
    lo = np.take_along_axis(lo, order, axis=1)
    hi = np.take_along_axis(hi, order, axis=1)
    run = np.maximum.accumulate(hi, axis=1)
    prev = np.concatenate([np.zeros((lo.shape[0], 1)), run[:, :-1]], axis=1)
    seg = np.maximum(lo - prev, 0.0)
    return seg.sum(axis=1)


def _volume(base_x, base_y):
    """Exact union volume V = integral_0^1 Lx(c)*Ly(c) dc (adaptive Simpson)."""
    i_idx = np.repeat(np.arange(_N), _N)
    j_idx = np.tile(np.arange(_N), _N)

    def f(c):
        return _union_len(base_x, i_idx) * _union_len(base_y, j_idx)

    def simp(h):
        n = int(round(1.0 / h))
        xs = np.linspace(0.0, 1.0, n + 1)
        vals = f(xs)
        return (h / 3.0) * (vals[0] + vals[-1] + 4.0 * vals[1:-1:2].sum()
                            + 2.0 * vals[2:-2:2].sum())

    v1 = simp(1.0 / 8.0)
    v2 = simp(1.0 / 16.0)
    v3 = simp(1.0 / 32.0)
    return v3 + (v3 - v2) / 7.0


def _keich_xy():
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return keich[i_grid.flatten()], keich[j_grid.flatten()]


def _make_obj(base_x, base_y):
    i_idx = np.repeat(np.arange(_N), _N)
    j_idx = np.tile(np.arange(_N), _N)

    def obj(v):
        return _volume(v[:_N], v[_N:])

    return obj


def _refine(v0, max_iter, max_time):
    """L-BFGS-B from v0; returns (best_v, best_f)."""
    obj = _make_obj(v0[:_N], v0[_N:])
    best_v, best_f = v0, obj(v0)
    res = minimize(obj, v0, method="L-BFGS-B",
                   options={"maxiter": max_iter, "maxfun": 4000,
                            "maxls": 40, "max_time": max_time})
    if res.fun < best_f:
        best_f = res.fun
        best_v = res.x
    return best_v, best_f


def _random_starts(rng, base_x, base_y, n):
    starts = []
    for _ in range(n):
        r = 0.12 * (0.3 + 0.7 * rng.random())
        bx = base_x + r * rng.standard_normal(_N)
        by = base_y + r * rng.standard_normal(_N)
        starts.append(np.concatenate([bx, by]))
    for _ in range(2):
        starts.append(np.concatenate([rng.uniform(-0.5, 0.5, _N),
                                      rng.uniform(-0.5, 0.5, _N)]))
    return starts


def kakeya_needle_3d8() -> np.ndarray:
    base_x, base_y = _keich_xy()
    seed_v = np.concatenate([base_x, base_y])

    best_v, best_f = _refine(seed_v, max_iter=300, max_time=120.0)

    rng = np.random.default_rng(12345)
    for v0 in _random_starts(rng, base_x, base_y, 24):
        if time.time() > _DEADLINE - _MARGIN:
            break
        v, f = _refine(v0, max_iter=150, max_time=40.0)
        if f < best_f:
            best_f = f
            best_v = v

    if time.time() < _DEADLINE - _MARGIN:
        v, f = _refine(best_v, max_iter=2000,
                       max_time=max(1.0, _DEADLINE - _MARGIN - 5.0))
        if f < best_f:
            best_f = f
            best_v = v

    bx = best_v[:_N]
    by = best_v[_N:]
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    positions = np.stack([bx[i_grid.flatten()], by[j_grid.flatten()]], axis=1)

    if not np.all(np.isfinite(positions)):
        return np.stack([base_x[i_grid.flatten()], base_y[j_grid.flatten()]], axis=1)
    return positions


# EVOLVE-BLOCK-END
