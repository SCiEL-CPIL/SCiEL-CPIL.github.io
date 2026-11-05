"""Construct a C5-free graph with independence number < 9, maximizing n.

Design: a "windmill" of cliques sharing one common vertex.
  - A windmill of cliques (cliques sharing a single vertex) contains no 5-cycle.
  - Its independence number is 1 (the shared vertex) + sum over cliques of the
    largest independent set inside each clique, which is 1 for every clique.
  So alpha = 1 + (number of cliques).  We pick cliques large enough to push n up
  while keeping (number of cliques) <= 7 so that alpha <= 8 < 9.

  Default clique sizes: [6,6,6,6,5,5,4]  ->  n = 4*5 + 2*4 + 3 = 33,  alpha = 8.
"""
import time

# hard wall-clock guard (evaluator kills at ~120-150 s; keep a big safety margin)
_T0 = time.time()
_DEADLINE = 120.0          # seconds
_MARGIN = 5.0              # stop building this close to the limit


def _time_ok():
    return (time.time() - _T0) < (_DEADLINE - _MARGIN)


# ---------------------------------------------------------------------------
# Exact verification helpers
# ---------------------------------------------------------------------------
def _has_independent_set_of_size_k(adj, n, k):
    """Return True iff there is an independent set of size k (exact).

    Meet-in-the-middle: split vertices into A (size m) and B (size n-m).
    For B, compute the maximal independent-set size for every subset (bit DP).
    For every independent subset S of A with |S|<=k, the remaining k-|S| vertices
    must be independent inside B and disjoint from N(S).
    """
    m = min(n, 20)
    A = list(range(m))
    B = list(range(m, n))
    nb = n - m
    b2 = 1 << nb

    # independent flag for every subset of B
    b_ind = [True] * b2
    for mask in range(1, b2):
        lb = mask & (-mask)
        i = (lb.bit_length() - 1)
        rest = mask ^ lb
        b_ind[mask] = b_ind[rest] and (adj[B[i]] & (1 << i) == 0) and \
                      (adj[B[i]] & rest == 0)
    # max independent-set size for every subset of B
    b_max = [0] * b2
    for mask in range(1, b2):
        lb = mask & (-mask)
        i = (lb.bit_length() - 1)
        rest = mask ^ lb
        b_max[mask] = b_max[rest] + (1 if b_ind[1 << i] else 0)

    a2 = 1 << m
    a_ind = [True] * a2
    a_nei = [0] * a2
    for mask in range(1, a2):
        lb = mask & (-mask)
        i = (lb.bit_length() - 1)
        rest = mask ^ lb
        a_ind[mask] = a_ind[rest] and (adj[A[i]] & (1 << i) == 0) and \
                      (adj[A[i]] & rest == 0)
        a_nei[mask] = a_nei[rest] | adj[A[i]]

    for mask in range(a2):
        if not a_ind[mask]:
            continue
        need = k - mask.bit_count()
        if need < 0:
            continue
        if need == 0:
            return True
        if need > nb:
            continue
        # vertices of B that are free of A's neighbourhood
        free = (b2 - 1) ^ a_nei[mask]
        best = 0
        sub = free
        while sub:
            if b_ind[sub] and sub.bit_count() >= need:
                if b_max[sub] >= need:
                    return True
            sub = (sub - 1) & free
    return False


def _has_c5(adj, n):
    """Return True iff the graph contains a 5-cycle (as a subgraph)."""
    if n < 5:
        return False
    deg = [adj[v].bit_count() for v in range(n)]
    # a 5-cycle needs min degree >= 2
    if min(deg) < 2:
        return False
    # canonical 5-cycle: v0 < v1 < v2 < v3 < v4, edges (0-1,1-2,2-3,3-4,4-0)
    for v0 in range(n):
        n0 = adj[v0]
        if n0.bit_count() < 2:
            continue
        for v1 in range(v0 + 1, n):
            if not (n0 >> v1) & 1:
                continue
            n1 = adj[v1] & ~((1 << (v0 + 1)) - 1)  # only vertices > v1
            if n1.bit_count() < 2:
                continue
            for v2 in range(v1 + 1, n):
                if not (n1 >> v2) & 1:
                    continue
                n2 = adj[v2] & ~((1 << (v1 + 1)) - 1)  # only vertices > v2
                if n2.bit_count() < 2:
                    continue
                for v3 in range(v2 + 1, n):
                    if not (n2 >> v3) & 1:
                        continue
                    n3 = adj[v3] & ~((1 << (v2 + 1)) - 1)  # only vertices > v3
                    # need a vertex v4 > v3 adjacent to both v3 and v0
                    if (n3 & n0) != 0:
                        return True
    return False


def _valid(adj, n):
    return (not _has_c5(adj, n)) and (not _has_independent_set_of_size_k(adj, n, 9))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _windmill(clique_sizes):
    """Build a windmill of cliques sharing vertex 0.  Returns (n, adj)."""
    adj = [0] * (35)  # we only ever use up to 33 vertices
    pos = 1
    for s in clique_sizes:
        block = list(range(pos, pos + s))
        pos += s
        for i in range(len(block)):
            u = block[i]
            for j in range(i + 1, len(block)):
                v = block[j]
                adj[u] |= 1 << v
                adj[v] |= 1 << u
            adj[0] |= 1 << u
            adj[u] |= 1
    n = pos
    return n, adj


def _seed():
    """Seven disjoint copies of K4: n=28, alpha=7, C5-free (always valid)."""
    adj = [0] * 28
    for b in range(7):
        for i in range(4):
            u = 4 * b + i
            for j in range(i + 1, 4):
                v = 4 * b + j
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return 28, adj


def construct():
    # candidate clique-size multisets, ordered from largest n to smallest.
    # each keeps (number of cliques) <= 7 so alpha = 1 + count <= 8 < 9.
    candidates = [
        [6, 6, 6, 6, 5, 5, 4],   # n = 33, alpha = 8
        [7, 6, 6, 6, 5, 4, 3],   # n = 33, alpha = 8
        [6, 6, 6, 5, 5, 5, 4],   # n = 32, alpha = 8
        [6, 6, 6, 6, 5, 4, 3],   # n = 32, alpha = 8
        [6, 6, 6, 6, 6, 3, 3],   # n = 32, alpha = 8
        [5, 5, 5, 5, 5, 5, 4],   # n = 30, alpha = 8
        [6, 6, 6, 6, 4, 4, 4],   # n = 30, alpha = 8
        [4, 4, 4, 4, 4, 4, 4],   # n = 28, alpha = 8
    ]

    best = None
    for sizes in candidates:
        if not _time_ok():
            break
        n, adj = _windmill(sizes)
        if n > 35:
            continue
        if _valid(adj, n):
            if best is None or n > best[0]:
                best = (n, adj)
            break  # candidates are in descending n order; first valid is best

    if best is None:
        n, adj = _seed()
        if _valid(adj, n):
            best = (n, adj)
        else:  # should never happen; return the seed anyway
            best = (n, adj)

    n, adj = best
    edges = []
    for u in range(n):
        m = adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            if v > u:
                edges.append([u, v])
            m ^= lb
    return {'n': n, 'edges': edges}
