import time
import numpy as np

_N = 8
_FRAC = 1.0 / _N
_IDX8 = np.arange(_N) / _N
# 1501 c-samples -> union length via vectorized interval-union; product -> volume
_C = np.linspace(0.0, 1.0, 1501)
_NN = _N * _N


def _union_len(C, base):
    shifts = C[:, None] * _IDX8[None, :]
    l = base[None, :] + shifts
    r = l + _FRAC
    order = np.argsort(l, axis=1)
    ls = np.take_along_axis(l, order, 1)
    rs = np.take_along_axis(r, order, 1)
    rs_cum = np.maximum.accumulate(rs, axis=1)
    gap = np.maximum(rs_cum[:, 1:] - ls[:, 1:], 0.0)
    return rs_cum[:, -1] - ls[:, 0] + gap.sum(axis=1)


def _vol(base):
    # base shape (64,2); separable product: V = mean_c (uf(c) * ug(c))
    x = base[:, 0]
    y = base[:, 1]
    # f_i = x_{8i+j} (i fixed, any j); g_j = y_{8i+j}
    f = x.reshape(_N, _N)[:, 0]
    g = y.reshape(_N, _N)[0, :]
    uf = _union_len(_C, f)
    ug = _union_len(_C, g)
    val = float(np.mean(uf * ug))
    if not np.isfinite(val):
        return 1e9
    return val


def _keich_v():
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N)
    for idx in range(N):
        bits = np.array(list(str(bin(idx))[2:]), dtype=int)
        bits = np.pad(bits, (pos - len(bits), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * bits * (2.0 ** -j_terms) / pos)
    keich = keich - keich[0]
    return keich


def _build(keich):
    f = np.concatenate([[0.0], keich[:7]])
    g = np.concatenate([[0.0], keich[7:14]])
    i = np.arange(_NN) // _N
    j = np.arange(_NN) % _N
    return np.stack([f[i], g[j]], axis=1)


def kakeya_needle_3d8() -> np.ndarray:
    keich = _keich_v()
    best = _build(keich).astype(float).copy()
    best_v = _vol(best)

    rng = np.random.default_rng(7)
    start = time.time()
    DEADLINE = start + 600.0 - 30.0  # safety margin

    T0 = 0.35
    T_end = 0.00003
    steps_per_run = 9000
    n_runs = 0
    while time.time() < DEADLINE:
        n_runs += 1
        # re-seed incumbent occasionally to escape deep basins
        if n_runs % 12 == 0:
            cur = best + rng.normal(0, 0.15, best.shape)
            cur_v = _vol(cur)
        else:
            cur = best.copy()
            cur_v = best_v

        T = T0
        for _ in range(steps_per_run):
            T *= 0.99965
            if T < T_end:
                break
            # pick a random tube and a random coordinate, propose a move
            k = rng.integers(0, _NN)
            d = rng.integers(0, 2)
            scale = 0.05 + 0.25 * (T / T0)
            delta = rng.normal(0.0, scale)
            old = cur[k, d]
            cur[k, d] = old + delta
            new_v = _vol(cur)
            if new_v < cur_v or (rng.random() < np.exp(-(new_v - cur_v) / max(T, 1e-12))):
                cur_v = new_v
                if new_v < best_v:
                    best_v = new_v
                    best = cur.copy()
            else:
                cur[k, d] = old

    if best.shape != (_NN, 2) or not np.all(np.isfinite(best)):
        best = _build(_keich_v())
    return best
