"""Algebraic construction of small 3-regular graphs of girth >= g.

Strategy (different family from random regular-graph search):
  - Use the Lazebnik-Ustimenko-Woldar (Woldar) product graph CD(d, q) with q = k.
    It is a simple k-regular graph on k^2 * (d+1) vertices.
  - Instead of a per-vertex BFS ball check, certify girth >= g exactly by counting
    non-backtracking closed walks of length < g: in any graph, a cycle of length L
    (L < g) exists iff there is a non-backtracking closed walk of length L, so the
    girth is >= g iff the number of non-backtracking closed walks of every length
    L in [3, g-1] is zero. This count is computed by matrix powers of a (k-1)*n
    non-backtracking matrix, which is far cheaper than n BFS balls of radius g/2-1.
  - Because the count is cheap, we can sweep many (d, q) parameter pairs (different
    vertex counts) and return the smallest valid one, instead of gambling on random
    regular graphs.
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges).

    d is chosen (as in the seed) to make the algebraic girth at least g.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _build_adjacency(k, g, d, q):
    """Build the adjacency list of CD(d, q) with a fixed d. Returns (n, adj) or (None, None)."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    if d < 2:
        return None, None
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
    n = len(order)
    adj = [[] for _ in range(n)]
    for vertex in order:
        vi = index[vertex]
        if vertex[0] == 0:
            for other in neighbours(vertex):
                adj[vi].append(index[other])
    return n, adj


def _girth_at_least_nb(adj, n, g):
    """Return True iff graph (adjacency list) has girth >= g.

    Exact: count non-backtracking closed walks of length L for L in [3, g-1].
    A cycle of length L exists iff a non-backtracking closed walk of length L exists.
    Uses the (k-1)*n non-backtracking matrix and matrix powers.
    """
    if g <= 3:
        return True
    if n == 0:
        return True
    k = len(adj[0])
    if k < 2:
        # degenerate: no 2-cycles possible; a simple graph has no 3-cycles if no triangle
        # just fall back to a direct triangle check for safety
        for i in range(n):
            for j in adj[i]:
                for l in adj[j]:
                    if l == i and j != l:
                        return False
        return True

    # Build non-backtracking matrix B of size (k-1)*n.
    # State (u, v) means "arrived at v from u". Transition to (v, w) for w in adj[v], w != u.
    # Row index of state (u, v): u*(k-1) + (rank of u in adj[v]).
    # We need, for each vertex v, the rank of each neighbor u in adj[v].
    rank = {}
    for v in range(n):
        for r, u in enumerate(adj[v]):
            rank[(u, v)] = r  # rank of u in adj[v]
    dim = (k - 1) * n
    B = np.zeros((dim, dim), dtype=np.int64)
    for v in range(n):
        for u in adj[v]:
            row = u * (k - 1) + rank[(u, v)]
            for w in adj[v]:
                if w == u:
                    continue
                col = v * (k - 1) + rank[(v, w)]
                B[row, col] = 1

    # Number of non-backtracking closed walks of length L = trace(B^L).
    # We need trace(B^L) == 0 for all L in [3, g-1].
    # Compute B^L iteratively, tracking trace.
    P = B.copy()
    # P currently B^1. We need from L=3.
    # B^2 = B @ B
    P2 = B @ B
    # We need trace for L in 3..g-1.
    cur = P2  # B^2
    for L in range(3, g):
        cur = cur @ B  # B^L
        tr = int(np.trace(cur))
        if tr != 0:
            return False
    return True


def _random_regular_graph(n, k, rng):
    """Generate a random simple k-regular graph on n vertices (n*k even).
    Returns adjacency list or None on failure."""
    if n * k % 2 != 0 or n < k + 1:
        return None
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    adj = [set() for _ in range(n)]
    for _attempt in range(200):
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, n * k, 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok and all(len(x) == k for x in adj):
            return [sorted(x) for x in adj]
    return None


def _deadline_reached(t0, limit):
    return (time.time() - t0) > limit


def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 138.0  # safety margin under 150s wall-clock

    # Moore bound lower bound on n
    if g >= 4:
        moore = 2 * (1 << (g // 2 - 1))
    else:
        moore = 0

    best = None
    best_n = None

    # Primary strategy: algebraic CD(d, q) sweep over d.
    # For q = k, the vertex count is k^2 * (d+1). Larger d -> larger n but larger girth.
    # The seed used a specific d; we sweep a range of d values to find the smallest n
    # whose girth is >= g. Because the girth check (non-backtracking count) is cheap,
    # we can afford to test many d values.
    # We want the smallest n, so we prefer small d, but d must be large enough for girth.
    # Sweep d from a low value upward; the first d with girth >= g gives a candidate.
    # Also sweep downward from the seed's d to find a smaller valid d.

    # Seed's d choice
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)

    # Candidate d values: try a window around d_seed, biased toward smaller d (smaller n).
    # n = k^2 * (d+1), so smaller d is better.
    # We'll test d values from max(2, d_seed - 12) up to d_seed + 4, but only those
    # that could plausibly give girth >= g. We test in order of increasing n (increasing d),
    # and keep the smallest n that passes. Actually we want the MINIMUM n that is valid,
    # so we should test d in increasing order and take the first valid one (smallest d).
    # But a smaller d may have girth < g. So we test all d in a range and pick the
    # smallest d that is valid.
    d_lo = max(2, d_seed - 14)
    d_hi = d_seed + 4

    tested = []
    for d in range(d_lo, d_hi + 1):
        if _deadline_reached(t0, HARD_LIMIT):
            break
        n_cand, adj = _build_adjacency(k, g, d, k)
        if n_cand is None:
            continue
        # Quick necessary condition: n must be at least the Moore bound.
        if n_cand < moore:
            continue
        # Girth check via non-backtracking closed walk count.
        if _girth_at_least_nb(adj, n_cand, g):
            edges = []
            for i in range(n_cand):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            if best_n is None or n_cand < best_n:
                best_n = n_cand
                best = (n_cand, edges)
        tested.append((d, n_cand))

    if best is not None:
        return best

    # Fallback: guaranteed-valid LUW construction (seed's d).
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    except Exception:
        pass

    # Last resort: random regular graph search (kept from seed for robustness).
    rng = random.Random(12345)
    step = 2 if k % 2 == 1 else 1
    lo = max(moore, k + 1)
    if lo % step:
        lo += step
    hi = lo + 200
    n = lo
    while n <= hi and not _deadline_reached(t0, HARD_LIMIT):
        max_attempts = 20 if n < 600 else 8
        for _ in range(max_attempts):
            if _deadline_reached(t0, HARD_LIMIT):
                break
            adj = _random_regular_graph(n, k, rng)
            if adj is None:
                continue
            if _girth_at_least_nb(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        n += step

    # Absolute fallback: raise (should not happen).
    raise RuntimeError("No valid construction found within time budget")
# EVOLVE-BLOCK-END
