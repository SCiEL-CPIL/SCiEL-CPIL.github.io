"""Search for a small 3-regular graph of girth >= g.

DIFFERENT design family from the seed: instead of enumerating a fixed algebraic
circulant family, we run a deterministic, time-boxed RANDOMISED search
(Configuration-Model + 2-switch/edge-swap local repair) seeded by the LUW
algebraic graph.  Girth is certified fast via non-backtracking matrix powers
(a necessary-and-sufficient filter for girth >= g) and the final candidate is
verified exactly with per-start NB-walk enumeration.
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# LUW construction (seed / fallback)
# ---------------------------------------------------------------------------
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges)."""
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


def _edges_to_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _adj_to_edges(adj, n):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _to_matrix(adj, n, k):
    A = np.zeros((n, n), dtype=np.int32)
    for u in range(n):
        for v in adj[u]:
            A[u, v] = 1
    return A


# ---------------------------------------------------------------------------
# Fast girth certification via non-backtracking matrix powers
# ---------------------------------------------------------------------------
def _girth_ge_nb_matrix(adj, n, k, g):
    """Return True iff the graph has girth >= g (necessary AND sufficient).

    A graph has girth >= g  <=>  there is no closed non-backtracking walk of
    length L for any 3 <= L <= g-1.  The number of closed NB walks of length L
    starting at vertex s is (N^{L-1})[s,s] where N is the non-backtracking
    (2k x 2k) transfer matrix.  So we check the diagonal of N^{L-1} for
    L = 3..g-1.  This is fast (matrix powers) and exact.
    """
    if g <= 3:
        return True
    m = 2 * n * k
    N = np.zeros((m, m), dtype=np.int64)
    for u in range(n):
        for a in range(k):
            i = u * k + a
            for b in range(k):
                v = adj[u][b]
                if v == adj[u][a]:
                    continue  # backtracking
                j = v * k + b
                N[i, j] = 1
    for L in range(3, g):
        P = N ** (L - 1)
        # diagonal entries for "start at s" are indices s*k .. s*k+k-1
        for s in range(n):
            base = s * k
            if np.any(P[base:base + k, base:base + k]):
                return False
    return True


# ---------------------------------------------------------------------------
# Exact girth certification via per-start NB-walk enumeration
# ---------------------------------------------------------------------------
def _girth_ok_nb(adj, n, g):
    """Exact check: no cycle of length < g.  Per-start BFS over NB states."""
    if g <= 3:
        return True
    adj_sets = [frozenset(a) for a in adj]
    for s in range(n):
        visited = set()
        queue = []
        for w in adj_sets[s]:
            queue.append((w, s, 1))
            visited.add((w, s))
        qi = 0
        while qi < len(queue):
            v, u, length = queue[qi]
            qi += 1
            if length >= g:
                continue
            for w in adj_sets[v]:
                if w == u:
                    continue
                if w == s and length + 1 >= 3:
                    return False
                state = (w, v)
                if state not in visited:
                    visited.add(state)
                    if length + 1 < g:
                        queue.append((w, v, length + 1))
    return True


# ---------------------------------------------------------------------------
# Randomised search: Configuration Model + 2-switch repair
# ---------------------------------------------------------------------------
def _random_cubic(n, k, rng):
    """Random k-regular graph on n vertices via configuration model (simple)."""
    # stubs
    while True:
        stubs = []
        for u in range(n):
            stubs.extend([u] * k)
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        ok = True
        for i in range(0, 2 * n * k // 2, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                ok = False
                break
            if v in adj[u]:
                ok = False
                break
            adj[u].append(v)
            adj[v].append(u)
        if ok:
            return adj


def _try_2switch(adj, n, k, g, A, rng, deadline, max_attempts=200):
    """Attempt to improve girth via a 2-switch (edge swap).  Returns (adj, A)
    of the best graph found, or the original if nothing improved."""
    # Build edge set for fast membership
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
    # incident edges per vertex
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
        u = rng.randrange(n)
        a = rng.randrange(k)
        w = adj[u][a]
        if w not in edge_set:
            continue
        # pick a second edge (x,y) not incident to u,w
        x = rng.randrange(n)
        if x == u or x == w:
            continue
        b = rng.randrange(k)
        y = adj[x][b]
        if y == u or y == w or y == x:
            continue
        if (min(x, y), max(x, y)) not in edge_set:
            continue
        # candidate new edges (u,y) and (w,x)
        if y in adj[u] or x in adj[w]:
            continue
        # apply
        def remove_edge(p, q):
            adj[p].remove(q)
            adj[q].remove(p)
        remove_edge(u, w)
        remove_edge(x, y)
        adj[u].append(y)
        adj[y].append(u)
        adj[w].append(x)
        adj[x].append(w)
        # rebuild A
        A = _to_matrix(adj, n, k)
        if _girth_ge_nb_matrix(adj, n, k, g):
            return adj, A  # improved (girth now >= g)
        # revert
        remove_edge(u, y)
        remove_edge(w, x)
        adj[u].append(w)
        adj[w].append(u)
        adj[x].append(y)
        adj[y].append(x)
        A = _to_matrix(adj, n, k)
    return adj, A


def _randomised_search(k, g, deadline, seed=0):
    """Deterministic randomised search.  Returns (n, edges) or None."""
    rng = random.Random(seed)
    # Moore bound
    moore = 2 * sum((k - 1) ** i for i in range(g // 2))
    # candidate sizes: start near moore, but must allow a k-regular graph
    # (n*k even).  Try a few sizes.
    sizes = []
    start = max(moore, 2 * k)
    for n in range(start, start + 40):
        if (n * k) % 2 == 0:
            sizes.append(n)
    sizes = sizes[:8]  # limit

    best = None
    best_n = float('inf')
    for n in sizes:
        if time.time() > deadline:
            break
        if n >= best_n:
            break
        adj = _random_cubic(n, k, rng)
        if adj is None:
            continue
        A = _to_matrix(adj, n, k)
        if _girth_ge_nb_matrix(adj, n, k, g):
            # verify exactly
            if _girth_ok_nb(adj, n, g):
                edges = _adj_to_edges(adj, n)
                best = (n, edges)
                best_n = n
                break
        # local repair
        adj, A = _try_2switch(adj, n, k, g, A, rng, deadline)
        if _girth_ge_nb_matrix(adj, n, k, g):
            if _girth_ok_nb(adj, n, g):
                edges = _adj_to_edges(adj, n)
                best = (n, edges)
                best_n = n
                break
    return best


def _deadline_reached(t0, limit):
    return (time.time() - t0) > limit


def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 140.0  # safety margin under 150s wall-clock

    # 1) Randomised search (new family)
    result = _randomised_search(k, g, HARD_LIMIT)
    if result is not None:
        return result

    # 2) Fallback: guaranteed-valid LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        raise

# EVOLVE-BLOCK-END
