import time
import numpy as np
from scipy import optimize

_N = 8
_FRAC = 1.0 / _N
_IDX8 = np.arange(_N) / _N
_C = np.linspace(0.0, 1.0, 101)


def _union_len(C, base):
    shifts = C[:, None] * _IDX8[None, :]
    l = base[None, :] + shifts
    r = l + _FRAC
    order = np.argsort(l, axis=1)
    ls = np.take_along_axis(l, order, 1)
    rs = np.take_along_axis(r, order, 1)
    union = (rs[:, 0] - ls[:, 0]).copy()
    cur = rs[:, 0:1]
    for k in range(1, _N):
        s = ls[:, k:k + 1]
        e = rs[:, k:k + 1]
        ne = np.maximum(cur, e)
        union = union + np.maximum(ne - np.maximum(cur, s), 0.0)
        cur = ne
    return union


def _v_to_fghk(v):
    f = np.zeros(_N)
    g = np.zeros(_N)
    h = np.zeros((_N, _N))
    k = np.zeros((_N, _N))
    f[1:] = v[:7]
    g[1:] = v[7:14]
    h = v[14:78].reshape(_N, _N)
    k = v[78:142].reshape(_N, _N)
    return f, g, h, k


def _obj_hybrid(v):
    f, g, h, k = _v_to_fghk(v)
    # This is a placeholder. We need to compute the full 3D volume.
    # For now, we return a dummy value
    return 1.0


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
    return np.concatenate([keich[1:], keich[1:]])


def _safe_minimize(func, x0, **kw):
    try:
        return optimize.minimize(func, x0, **kw)
    except Exception:
        return None


def kakeya_needle_3d8() -> np.ndarray:
    N = _N
    # Use the separable solution as a starting point
    keich = _keich_v()
    f = np.concatenate([[0.0], keich[:7]])
    g = np.concatenate([[0.0], keich[7:14]])
    
    # Build the hybrid parameter vector
    h = np.zeros((N, N))
    k = np.zeros((N, N))
    v0 = np.concatenate([f[1:], g[1:], h.flatten(), k.flatten()])
    
    best_v = v0.copy()
    # best_val = _obj_hybrid(v0) # Not implemented yet
    
    # For now, return the separable solution
    i_idx = np.arange(N * N) // N
    j_idx = np.arange(N * N) % N
    pos = np.stack([f[i_idx], g[j_idx]], axis=1)
    
    return pos
