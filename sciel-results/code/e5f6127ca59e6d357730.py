import time
import math
from collections import deque

_DEADLINE = 120.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


# ---------------------------------------------------------------------------
# Non-backtracking girth check (exact, up to a depth cap).
# ---------------------------------------------------------------------------
def _girth(adj, n, limit):
    INF = 10**9
    best = INF
    for u in range(n):
        for v in adj[u]:
            if u > v:
                continue
            dist = {}
            q = deque()
            for w in adj[v]:
                if w != u:
                    st = (v, w)
                    dist[st] = 2
                    q.append(st)
            while q:
                prev, cur = q.popleft()
                d = dist[(prev, cur)]
                if cur == u:
                    cyc = d
                    if cyc < best:
                        best = cyc
                        if best <= limit:
                            q.clear()
                            break
                    continue
                if d >= limit:
                    continue
                for nb in adj[cur]:
                    if nb != prev:
                        st = (cur, nb)
                        if st not in dist:
                            dist[st] = d + 1
                            q.append(st)
    return best if best < INF else limit + 1


def _build_adj(n, edges, k):
    adj = [[] for _ in range(n)]
    eset = set()
    for a, b in edges:
        if not (0 <= a < n and 0 <= b < n) or a == b:
            return None
        key = (a, b) if a < b else (b, a)
        if key in eset:
            return None
        eset.add(key)
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    return adj


def _verify(adj, n, k, g):
    # connectivity
    seen = [False] * n
    seen[0] = True
    st = [0]
    cnt = 1
    while st:
        v = st.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                st.append(w)
    if cnt != n:
        return False
    girth = _girth(adj, n, g - 1)
    return girth >= g


def _edges_from_adj(adj):
    e = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                e.append((u, v))
    return e


# ---------------------------------------------------------------------------
# Canonical Lazebnik-Ustimenko-Woldar generator (balanced / "symmetric" form).
# Produces a k-regular graph on n = k * 2^d vertices with girth >= 2d.
#
# Vertices: (side, y[0..d-1]) with side in {0,1}, y_i in Z_k.
# For a vertex on side 0, its neighbours are on side 1 with
#   y[0] = first,  y[i] = y[i-1] * x[i-1]  (x = the vertex's own y, x[0] unused)
# For side 1, neighbours are on side 0 with the same rule (substitution = add).
# ---------------------------------------------------------------------------
def _luw_balanced(k, d):
    # neighbour generation
    def neighbours(vertex):
        side, x = vertex  # x is a tuple of length d
        for first in range(k):
            y = [None] + [first] + [0] * (d - 1)
            for c in range(2, d + 1):
                y[c] = (y[c - 1] * x[c - 1]) % k
            yield (1 - side), tuple(y[1:])

    index = {(0, tuple([0] * d)): 0}
    order = [(0, tuple([0] * d))]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _luw_fallback(k, g):
    # girth >= 2d  =>  need 2d >= g  =>  d = ceil(g/2)
    d = math.ceil(g / 2)
    if d < 3:
        d = 3
    return _luw_balanced(k, d)


def _try_repair(n, edges, k, g, budget):
    """Attempt to shorten a verified k-regular girth>=g graph via 2-opt style
    edge swaps that preserve degree, regularity, simple-ness and girth.
    Always returns the best *verified* graph found (possibly the input)."""
    adj = _build_adj(n, edges, k)
    if adj is None or not _verify(adj, n, k, g):
        return n, edges

    best_n = n
    best_edges = edges
    best_adj = adj
    # candidate edge list for swaps
    cand = list(edges)

    def try_swap(e1, e2):
        nonlocal best_n, best_edges, best_adj
        (a, b), (c, d) = e1, e2
        # swap: remove (a,b),(c,d); add (a,c),(b,d) or (a,d),(b,c)
        for x, y, u, v in ((a, c, b, d), (a, d, b, c)):
            if x == y or u == v:
                continue
            # must be new edges
            if y in adj[x] or v in adj[u]:
                continue
            # apply
            adj[x].remove(y)
            adj[c].remove(d)
            adj[x].append(u)
            adj[c].append(v)
            adj[b].remove(a)
            adj[d].remove(c)
            adj[b].append(v)
            adj[d].append(a)
            # quick girth pre-check only up to g-1
            ok = _verify(adj, n, k, g)
            if not ok:
                # revert
                adj[x].remove(u)
                adj[c].remove(v)
                adj[x].append(y)
                adj[c].append(d)
                adj[b].remove(v)
                adj[d].remove(a)
                adj[b].append(a)
                adj[d].append(c)
            return ok
        return False

    import random
    rng = random.Random(12345)
    deadline = time.time() + budget
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        rng.shuffle(cand)
        for i in range(len(cand)):
            if time.time() > deadline:
                break
            e1 = cand[i]
            for j in range(i + 1, min(i + 40, len(cand))):
                if time.time() > deadline:
                    break
                e2 = cand[j]
                # share a vertex? swapping two adjacent edges changes degree -> skip
                if (e1[0] in e2) or (e1[1] in e2):
                    continue
                if try_swap(e1, e2):
                    # accepted; rebuild edge list (still n vertices)
                    cand = _edges_from_adj(adj)
                    improved = True
                    break
    return n, _edges_from_adj(best_adj)


def construct(k, g):
    if k == 4 and g == 10:
        # Guaranteed-verified baseline: balanced L-U-W with d=5 -> 2048 vertices, girth>=10.
        n, edges = _luw_balanced(k, 5)
        adj = _build_adj(n, edges, k)
        if adj is None or not _verify(adj, n, k, g):
            n, edges = _luw_fallback(k, g)
        # Spend remaining time trying to shorten, always keeping a valid graph.
        left = _time_left()
        if left > 25 and n > 512:
            n, edges = _try_repair(n, edges, k, g, left - 20)
        return n, edges

    # General fallback.
    return _luw_fallback(k, g)
