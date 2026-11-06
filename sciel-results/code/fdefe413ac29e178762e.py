"""
Q3-free graph on 16 vertices, maximizing edges.

STRUCTURAL CHANGE: Precompute every Q3 embedding as a 120-bit mask over the 120
possible edges.  A graph contains a Q3 iff there is a forbidden mask whose all
12 edges are present in the graph.  Adding an edge e creates a Q3 iff e belongs
to some forbidden mask whose other 11 edges are all already present.  We
precompute, per edge, the list of forbidden masks containing it, so both the
incremental "does adding e create Q3" test and the full "is graph Q3-free" test
reduce to a small number of 120-bit AND/POPCOUNT operations (numpy).  This
replaces the old permutation-enumeration checks (hundreds of thousands of set
lookups per test) with O(1) bit operations, enabling far more greedy passes and
local-search moves within the time budget.
"""

import time
import random
import numpy as np

_DEADLINE = 130.0
_start = None

def _time_left():
    global _start
    if _start is None:
        _start = time.time()
    return _DEADLINE - (time.time() - _start)

def _timeup():
    return _time_left() <= 0

N = 16
_edges = []
_eidx = {}
_k = 0
for i in range(N):
    for j in range(i + 1, N):
        _eidx[(i, j)] = _k
        _edges.append((i, j))
        _k += 1
M = _k  # 120

_q3canon = []
for i in range(8):
    for j in range(i + 1, 8):
        if bin(i ^ j).count('1') == 1:
            _q3canon.append((i, j))
assert len(_q3canon) == 12

def _build_patterns():
    from itertools import permutations
    pats = set()
    for perm in permutations(range(8)):
        es = []
        for (i, j) in _q3canon:
            a = perm[i]
            b = perm[j]
            if a > b:
                a, b = b, a
            es.append(_eidx[(a, b)])
        pats.add(frozenset(es))
    return pats

_pats = _build_patterns()
_npats = len(_pats)
_edge_to_pats = [[] for _ in range(M)]
for pi, p in enumerate(_pats):
    for ei in p:
        _edge_to_pats[ei].append(pi)

_PAT_ARR = np.zeros((_npats, M), dtype=np.uint8)
for pi, p in enumerate(_pats):
    for ei in p:
        _PAT_ARR[pi, ei] = 1
# Precompute per-pattern "all-but-e" masks for fast incremental test
# For each edge ei and each pattern pi containing ei: mask of the other 11 edges
_pat_other = {}  # (ei, pi) -> np.uint8 array (other 11 edges)
for pi, p in enumerate(_pats):
    for ei in p:
        others = np.zeros(M, dtype=np.uint8)
        for oj in p:
            if oj != ei:
                others[oj] = 1
        _pat_other[(ei, pi)] = others

def _is_q3_free(mask):
    """True iff no forbidden pattern is fully contained in mask."""
    # A pattern is present if all 12 of its edges are in mask.
    # count of pattern edges present == 12.
    cnt = np.sum(_PAT_ARR & mask, axis=1)
    return bool(np.all(cnt < 12))

def _add_creates_q3(mask, ei):
    """True iff adding edge ei to mask creates a Q3."""
    if mask[ei]:
        return False
    for pi in _edge_to_pats[ei]:
        others = _pat_other[(ei, pi)]
        if int(np.sum(others & mask)) == 11:
            return True
    return False

def construct():
    global _start
    _start = time.time()
    random.seed(1234)

    def mask_to_edges(mask):
        return [[u, v] for (u, v) in _edges if mask[_eidx[(u, v)]]]

    def greedy(mask, iters=40):
        for _ in range(iters):
            if _timeup():
                break
            order = list(range(M))
            random.shuffle(order)
            improved = False
            for ei in order:
                if mask[ei]:
                    continue
                if not _add_creates_q3(mask, ei):
                    mask[ei] = 1
                    improved = True
            if not improved:
                break
        return mask

    def local_search(mask, budget=3.0):
        t_end = _start + budget
        cur = mask.copy()
        best = cur.copy()
        best_cnt = int(cur.sum())
        while time.time() < t_end and not _timeup():
            order = list(range(M))
            random.shuffle(order)
            for ei in order:
                if not cur[ei]:
                    continue
                cur[ei] = 0
                order2 = list(range(M))
                random.shuffle(order2)
                added = False
                for ej in order2:
                    if ej == ei or cur[ej]:
                        continue
                    if not _add_creates_q3(cur, ej):
                        cur[ej] = 1
                        added = True
                        break
                if not added:
                    cur[ei] = 1
                else:
                    for ej in order2:
                        if cur[ej]:
                            continue
                        if not _add_creates_q3(cur, ej):
                            cur[ej] = 1
                            break
                cnt = int(cur.sum())
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = cur.copy()
        return best

    best_mask = None
    best_cnt = -1

    def consider(mask):
        nonlocal best_mask, best_cnt
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_mask = mask.copy()

    seeds = []
    m0 = np.zeros(M, dtype=np.uint8)
    for a in range(3):
        for b in range(3, 16):
            m0[_eidx[(a, b)]] = 1
    seeds.append(m0)
    for p in (0.5, 0.6, 0.7):
        mr = np.zeros(M, dtype=np.uint8)
        for ei in range(M):
            if random.random() < p:
                mr[ei] = 1
        seeds.append(mr)

    for s in seeds:
        if _timeup():
            break
        m = s.copy()
        m = greedy(m)
        consider(m)
        m = local_search(m, budget=3.0)
        consider(m)
        m = greedy(m, iters=10)
        consider(m)

    for _ in range(40):
        if _timeup():
            break
        mr = np.zeros(M, dtype=np.uint8)
        for ei in range(M):
            if random.random() < 0.6:
                mr[ei] = 1
        mr = greedy(mr)
        consider(mr)
        mr = local_search(mr, budget=2.0)
        consider(mr)
        mr = greedy(mr, iters=8)
        consider(mr)

    if best_mask is None:
        best_mask = np.zeros(M, dtype=np.uint8)

    # Final safety: verify best is Q3-free
    if not _is_q3_free(best_mask):
        # fallback: strip edges until free
        order = list(range(M))
        random.shuffle(order)
        for ei in order:
            if best_mask[ei] and not _is_q3_free(best_mask):
                best_mask[ei] = 0

    return mask_to_edges(best_mask)
