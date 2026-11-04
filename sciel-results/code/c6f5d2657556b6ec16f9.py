"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11
SUP_SIZE = 4
N_SUP = 330
MAX_MASK = (1 << D) - 1  # 2047
N_SIGNS = 1 << SUP_SIZE  # 16


def _build_supports():
    sups = np.empty(N_SUP, dtype=np.int64)
    k = 0
    for comb in combinations(range(D), SUP_SIZE):
        m = 0
        for idx in comb:
            m |= (1 << idx)
        sups[k] = m
        k += 1
    return sups


SUPPORTS = _build_supports()
BIT_COUNT_LUT = np.array([i.bit_count() for i in range(MAX_MASK + 1)], dtype=np.uint8)

# Conflict graph: supports i,j conflict iff they share >=3 coordinates (intersection > 2)
_pair = (SUPPORTS[:, None] & SUPPORTS[None, :])
CONFLICT = (BIT_COUNT_LUT[_pair.astype(np.intp)] > 2)
# Compatibility mask (excluding self): supports that can coexist with row i
COMPAT = (~CONFLICT)
np.fill_diagonal(COMPAT, False)
COMPAT = COMPAT.astype(np.int8)
DEG0 = COMPAT.sum(axis=1).astype(np.int32)


def _greedy_mis(rng):
    """Min-degree greedy independent set with a live open set and degree array."""
    n = N_SUP
    open_set = np.ones(n, dtype=bool)
    deg = DEG0.copy()

    while open_set.any():
        open_idx = np.nonzero(open_set)[0]
        # min degree among open nodes, random tie-break
        dmin = deg[open_idx].min()
        cands = open_idx[deg[open_idx] == dmin]
        v = int(cands[rng.integers(len(cands))])
        # delete v and all its open neighbors
        nb = np.nonzero(CONFLICT[v] & open_set)[0]
        open_set[v] = False
        if nb.size:
            open_set[nb] = False
            deg[nb] -= 1
    return open_set


def _local_search(sel, rng, deadline):
    """Improve an independent set with vectorized 1-swap and 2-for-1 moves."""
    n = N_SUP
    sel = sel.copy()
    cur_size = int(sel.sum())
    for _ in range(400):
        if time.time() > deadline - 6.0:
            break
        unsel = np.nonzero(~sel)[0]
        if unsel.size == 0:
            break
        # conflicts of each unselected support with the current selected set
        cw = COMPAT[unsel].dot(sel.astype(np.int8))
        # 2-for-1: add u (conflicting with <=2 selected) while dropping those
        c2 = unsel[cw <= 2]
        if c2.size:
            u = int(c2[rng.integers(c2.size)])
            bad = np.nonzero(CONFLICT[u] & sel)[0]
            if bad.size < cur_size:  # strictly improves count
                sel[bad] = False
                sel[u] = True
                cur_size = int(sel.sum())
                continue
        # 1-swap: add u (conflicting with exactly 1 selected) while dropping it
        c1 = unsel[cw == 1]
        if c1.size:
            u = int(c1[rng.integers(c1.size)])
            bad = np.nonzero(CONFLICT[u] & sel)[0]
            sel[bad[0]] = False
            sel[u] = True
            continue
        # random 1-swap to escape local minima
        s_idx = np.nonzero(sel)[0]
        if s_idx.size == 0 or unsel.size == 0:
            break
        v = int(s_idx[rng.integers(s_idx.size)])
        u = int(unsel[rng.integers(unsel.size)])
        if not bool(CONFLICT[u, v]):  # u is compatible with everything selected except v
            sel[v] = False
            sel[u] = True
    return sel


def construct():
    start = time.time()
    deadline = start + 145.0

    best_sel = np.zeros(N_SUP, dtype=bool)
    best_size = 0
    rng = np.random.default_rng(2024)

    while True:
        if time.time() > deadline - 30.0:
            break
        sel = _greedy_mis(rng)
        size = int(sel.sum())
        if size > best_size:
            best_size = size
            best_sel = sel.copy()
        if time.time() > deadline - 25.0:
            break
        sel2 = _local_search(sel, rng, deadline)
        size2 = int(sel2.sum())
        if size2 > best_size:
            best_size = size2
            best_sel = sel2.copy()

    # Build vectors: 16 sign variants per selected support
    idxs = np.nonzero(best_sel)[0]
    if idxs.size == 0:
        return np.zeros((0, D), dtype=np.int64)
    masks = SUPPORTS[idxs]
    vecs = np.zeros((idxs.size * N_SIGNS, D), dtype=np.int64)
    smasks = np.arange(N_SIGNS)[:, None]  # (16,1)
    for b in range(D):
        bit = (masks[:, None] >> b) & 1          # (S,1)
        pos = (smasks & (1 << b)) != 0           # (16,1)
        vals = np.where(bit, np.where(pos, 1, -1), 0)  # (S,16)
        vecs[:, b] = vals.reshape(-1)

    # Vectorized safety check: min pairwise distance >= max norm
    n = vecs.shape[0]
    norms_sq = (vecs ** 2).sum(axis=1)
    max_norm_sq = int(norms_sq.max())
    if n > 1:
        dots = vecs @ vecs.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, 0)
        if int(dist_sq.min()) < max_norm_sq:
            # drop violating rows (should not happen for valid supports)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                bad = np.nonzero((dist_sq[i] < max_norm_sq) & keep)[0]
                bad = bad[bad > i]
                keep[bad] = False
            vecs = vecs[keep]

    return vecs
