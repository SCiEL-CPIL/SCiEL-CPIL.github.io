"""
Exhaustive search over all valid subsets of the 17-vertex graph,
maximizing the number of distinct degrees in the induced subgraph.
"""
import time
from itertools import combinations

P = {
    'n': 17,
    'edges': [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16],
        [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16],
        [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15],
        [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16],
        [4, 5], [4, 6], [4, 8], [4, 12], [4, 13],
        [5, 6], [5, 7], [5, 9], [5, 13], [5, 14],
        [6, 7], [6, 8], [6, 10], [6, 14], [6, 15],
        [7, 8], [7, 9], [7, 11], [7, 15], [7, 16],
        [8, 9], [8, 10], [8, 12], [8, 16],
        [9, 10], [9, 11], [9, 13],
        [10, 11], [10, 12], [10, 14],
        [11, 12], [11, 13], [11, 15],
        [12, 13], [12, 14], [12, 16],
        [13, 14], [13, 15],
        [14, 15], [14, 16],
        [15, 16]
    ],
    'homogeneous_cap': 4,
    'minimum_vertices': 8
}

_DEADLINE = 85.0  # seconds, safety margin from 90s kill


def _adjacency_bitmask():
    n = P['n']
    bm = [0] * n
    for a, b in P['edges']:
        bm[a] |= (1 << b)
        bm[b] |= (1 << a)
    return bm


def _evaluate_subset(subset, adj_bm):
    """Return the number of distinct degrees in the induced subgraph."""
    mask = 0
    for v in subset:
        mask |= (1 << v)
    degs = set()
    for v in subset:
        d = (adj_bm[v] & mask).bit_count()
        degs.add(d)
    return len(degs)


def _find_best():
    n = P['n']
    min_v = P['minimum_vertices']
    adj_bm = _adjacency_bitmask()
    t0 = time.time()

    best_subset = None
    best_nd = 0

    # Try all subset sizes from n down to min_v
    for k in range(n, min_v - 1, -1):
        if time.time() - t0 > _DEADLINE:
            break
        for comb in combinations(range(n), k):
            if time.time() - t0 > _DEADLINE:
                return best_subset
            nd = _evaluate_subset(comb, adj_bm)
            if nd > best_nd:
                best_nd = nd
                best_subset = list(comb)
        # If we found k distinct degrees in a k-vertex subset, that's optimal
        if best_nd == k:
            break

    return best_subset


# Precompute at module load
_BEST = _find_best()


def construct():
    if _BEST is None:
        return list(range(P['minimum_vertices']))
    result = [int(v) for v in _BEST]
    if len(result) < P['minimum_vertices']:
        result = list(range(P['minimum_vertices']))
    return result
