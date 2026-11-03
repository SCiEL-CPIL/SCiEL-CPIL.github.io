"""Kissing number in dimension 11: integer vectors with min pairwise dist >= max norm.

Strategy: exploit the Johnson association scheme J(11,4).
- Candidate pool = weight-4 vectors (norm 2, M^2 = 4). Conflict = share >= 2 coords.
- Add weight-3 vectors (norm^2=3) which must be pairwise disjoint and share <=1 coord
  with every chosen weight-4 vector.
- Greedy min-degree construction + 1-for-2 local search on the conflict graph.
"""
import time
import numpy as np
from itertools import combinations

D = 11
TIME_LIMIT = 150.0


def _make_weight_vectors(max_weight):
    vecs = []
    for w in range(1, max_weight + 1):
        for idx in combinations(range(D), w):
            for signs in range(1 << w):
                v = [0] * D
                for j, i in enumerate(idx):
                    v[i] = 1 if (signs >> j) & 1 else -1
                vecs.append(v)
    return vecs


def _conflict_w4(w4):
    n4 = len(w4)
    w4s = np.array(w4, dtype=np.int8)
    conflict = np.zeros((n4, n4), dtype=bool)
    coord_bits = [np.zeros(n4, dtype=np.uint64) for _ in range(D)]
    for i in range(n4):
        for j in range(D):
            if w4s[i, j]:
                coord_bits[j][i] = 1
    for a in range(n4):
        counts = np.zeros(n4, dtype=np.int32)
        for j in range(D):
            if w4s[a, j]:
                counts += coord_bits[j]
        conflict[a] = counts >= 2
    return conflict | conflict.T


def _conflict_w3(w3):
    n3 = len(w3)
    if n3 == 0:
        return np.zeros((0, 0), dtype=bool)
    w3s = np.array(w3, dtype=np.int8)
    conflict = np.zeros((n3, n3), dtype=bool)
    coord_bits = [np.zeros(n3, dtype=np.uint64) for _ in range(D)]
    for i in range(n3):
        for j in range(D):
            if w3s[i, j]:
                coord_bits[j][i] = 1
    for a in range(n3):
        counts = np.zeros(n3, dtype=np.int32)
        for j in range(D):
            if w3s[a, j]:
                counts += coord_bits[j]
        conflict[a] = counts >= 1
    return conflict | conflict.T


def _valid_w3(w3, w4sel):
    if not w4sel:
        return list(range(len(w3)))
    w4s = np.array(w4sel, dtype=np.int8)
    w3s = np.array(w3, dtype=np.int8)
    coord_bits = [np.zeros(len(w4sel), dtype=np.uint64) for _ in range(D)]
    for i in range(len(w4sel)):
        for j in range(D):
            if w4s[i, j]:
                coord_bits[j][i] = 1
    valid = []
    for a in range(len(w3)):
        counts = np.zeros(len(w4sel), dtype=np.int32)
        for j in range(D):
            if w3s[a, j]:
                counts += coord_bits[j]
        if not np.any(counts >= 2):
            valid.append(a)
    return valid


def _greedy_independent(conflict, deg0, rng=None):
    n = len(deg0)
    deg = deg0.copy()
    chosen = set()
    for _ in range(n):
        avail = np.array([i for i in range(n) if i not in chosen])
        if len(avail) == 0:
            break
        dmin = deg[avail].min()
        cand = avail[deg[avail] == dmin]
        if rng is not None:
            i = cand[rng.integers(0, len(cand))]
        else:
            i = cand[0]
        chosen.add(i)
        neigh = conflict[i]
        non_neigh = ~neigh
        non_neigh[i] = False
        not_chosen = ~np.array([c in chosen for c in range(n)])
        deg[non_neigh & not_chosen] -= 1
        deg[i] = -1
    return chosen


def _local_search(chosen, conflict, max_iters=3000, deadline=None):
    chosen = set(chosen)
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        if deadline is not None and time.time() > deadline:
            break
        for u in range(len(conflict)):
            if u in chosen:
                continue
            neigh = conflict[u]
            ch_neigh = [v for v in chosen if neigh[v]]
            if len(ch_neigh) >= 2:
                a, b = ch_neigh[0], ch_neigh[1]
                if not conflict[a, b]:
                    chosen.discard(a)
                    chosen.discard(b)
                    chosen.add(u)
                    improved = True
                    break
    return chosen


def construct():
    t0 = time.time()
    w4 = _make_weight_vectors(4)
    w3 = _make_weight_vectors(3)
    n4 = len(w4)
    conflict4 = _conflict_w4(w4)
    conflict3 = _conflict_w3(w3)
    deg4 = conflict4.sum(axis=1)

    # initial greedy
    best4 = _greedy_independent(conflict4, deg4)
    best_count = len(best4)

    # restarts
    restarts = 0
    while time.time() - t0 < TIME_LIMIT - 30 and restarts < 80:
        restarts += 1
        rng = np.random.default_rng(restarts * 12345 + 7)
        c4 = _greedy_independent(conflict4, deg4, rng)
        c4 = _local_search(c4, conflict4, deadline=TIME_LIMIT - 25 + t0)
        if len(c4) > best_count:
            best_count = len(c4)
            best4 = c4

    best4 = sorted(best4)
    # add weight-3
    valid_w3 = _valid_w3(w3, [w4[i] for i in best4])
    vw3 = np.array(valid_w3)
    if len(vw3) > 0:
        cw3 = conflict3[np.ix_(vw3, vw3)]
        d3 = cw3.sum(axis=1)
        chosen3 = _greedy_independent(cw3, d3)
        chosen3 = _local_search(chosen3, cw3, deadline=TIME_LIMIT - 15 + t0)
        chosen3 = [int(vw3[i]) for i in sorted(chosen3)]
    else:
        chosen3 = []

    result = []
    for i in best4:
        result.append(w4[i])
    for i in chosen3:
        result.append(w3[i])

    if not result:
        result = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    result.append(v)

    arr = np.array(result, dtype=np.int64)

    # fast vectorized validation: min pairwise dist^2 >= max norm^2
    if len(arr) > 1 and time.time() - t0 < TIME_LIMIT - 3:
        norms2 = (arr ** 2).sum(axis=1)
        max_norm2 = int(norms2.max())
        # compute all pairwise dist^2 via chunked gram
        n = len(arr)
        min_d2 = max_norm2  # assume ok, verify
        # dist^2(i,j) = ||a_i||^2 + ||a_j||^2 - 2 a_i.a_j
        G = arr @ arr.T
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            block = G[start:stop, :]
            d2 = norms2[start:stop, None] + norms2[None, :] - 2 * block
            # exclude diagonal
            rows = np.arange(stop - start)
            cols = np.arange(n)
            d2[rows, start + rows] = np.inf
            m = d2.min()
            if m < min_d2:
                min_d2 = int(m)
        if min_d2 < max_norm2:
            # invalid: fall back to seed (weight-2) which is guaranteed valid
            result = []
            for i, j in combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        result.append(v)
            arr = np.array(result, dtype=np.int64)

    return arr
