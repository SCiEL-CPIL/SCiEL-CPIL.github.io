"""Rewritten program: exact maximum-diversity selection over the pinned 17-vertex host.

Approach (built on the "diversity selection" idea, but made exact/stronger):
  * Each vertex v has a degree signature relative to a candidate set S, namely
    deg_S(v) = |N(v) ∩ S| (degree of v inside the induced subgraph on S).
  * The objective is the number of *distinct* values among {deg_S(v) : v in S},
    with |S| >= 8.
  * A multi-seed greedy diversity heuristic (iteratively add the vertex that
    creates the most new distinct degree values) is run to get a fast, good
    starting solution and to honor the diversity-selection framing.
  * Because the host has only 17 vertices, the space of all subsets with
    |S| >= 8 is tiny (~90k masks).  We therefore also perform an *exhaustive*
    enumeration over every such subset, which provably returns the global
    optimum and strictly dominates any heuristic.  The final answer is the best
    of the two (the exhaustive optimum).

The host property (no K4, no independent set of size 4) is a fixed property of
the given graph and is checked by the evaluator, not something we must enforce.
"""
import random

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_MINV = P['minimum_vertices']

# Precompute the adjacency of every vertex as a 17-bit mask.
_ADJ = [0] * _N
for _a, _b in P['edges']:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)


def _distinct_deg(mask):
    """Number of distinct induced-subgraph degrees among the vertices in `mask`."""
    degs = set()
    v = 0
    m = mask
    while m:
        if m & 1:
            degs.add((_ADJ[v] & mask).bit_count())
        m >>= 1
        v += 1
    return len(degs)


def _greedy_diversity(rng):
    """Multi-seed greedy: start from 8 random vertices, then repeatedly add the
    vertex that maximises the number of *new* distinct degree values."""
    verts = list(range(_N))
    rng.shuffle(verts)
    chosen = verts[:_MINV]
    chosen_set = set(chosen)
    cur_mask = 0
    for v in chosen:
        cur_mask |= (1 << v)
    cur_d = _distinct_deg(cur_mask)
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_v = -1
        for v in range(_N):
            if v in chosen_set:
                continue
            new_mask = cur_mask | (1 << v)
            d = _distinct_deg(new_mask)
            gain = d - cur_d
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v >= 0 and best_gain > 0:
            chosen_set.add(best_v)
            chosen.append(best_v)
            cur_mask |= (1 << best_v)
            cur_d += best_gain
            improved = True
    return cur_mask, cur_d


def _exhaustive_best():
    """Exact optimum: scan every subset with popcount >= _MINV."""
    best_d = -1
    best_mask = 0
    for mask in range(1 << _N):
        if mask.bit_count() < _MINV:
            continue
        d = _distinct_deg(mask)
        if d > best_d:
            best_d = d
            best_mask = mask
    return best_mask, best_d


def _mask_to_vertices(mask):
    out = []
    v = 0
    m = mask
    while m:
        if m & 1:
            out.append(v)
        m >>= 1
        v += 1
    return out


def construct():
    # Deterministic random seeds for the greedy diversity heuristic.
    rng = random.Random(20240517)
    greedy_mask, greedy_d = -1, -1
    for _ in range(400):
        m, d = _greedy_diversity(rng)
        if d > greedy_d or (d == greedy_d and m.bit_count() > greedy_mask.bit_count()):
            greedy_mask, greedy_d = m, d

    # Exact global optimum over the (tiny) space of all valid subsets.
    ex_mask, ex_d = _exhaustive_best()

    # Take the better of the two (exhaustive is provably optimal).
    if ex_d >= greedy_d:
        final_mask = ex_mask
    else:
        final_mask = greedy_mask

    verts = _mask_to_vertices(final_mask)
    # Defensive: guarantee at least _MINV distinct Python-int vertices.
    if len(verts) < _MINV:
        extra = [v for v in range(_N) if v not in set(verts)]
        verts = verts + extra[: _MINV - len(verts)]
    verts = [int(v) for v in sorted(verts)]
    return verts
# EVOLVE-BLOCK-END
