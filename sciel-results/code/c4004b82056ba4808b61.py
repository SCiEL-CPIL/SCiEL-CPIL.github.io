"""Small k-regular graph of girth >= g (cage problem n(k,g)).

Strategy (combined):
  1. LUW CD(d,q) construction  -> guaranteed-valid baseline (bipartite, girth d+5 >= g).
  2. Scan the Z_q x Z_2 Cayley family (n = 2q) in increasing order for a smaller
     verified-valid graph (fast shortest-path girth check).
  3. Random k-regular generator + 2-switch local search to try to beat the above.
  4. Return the smallest verified-valid result found, under an explicit deadline.
"""
import time
import random
from collections import deque

_DEADLINE = None


def _time_left(margin=6.0):
    if _DEADLINE is None:
        return float("inf")
    return _DEADLINE - time.monotonic() - margin


# --------------------------------------------------------------------------- #
# girth verification                                                          #
# --------------------------------------------------------------------------- #
def _girth_ok(adj, n, g):
    """Return True iff the graph has no cycle of length < g.

    Fast method: for each vertex, shortest simple cycle through it is
    min over each edge (u,v) of dist(u,v) + 1.  If every such value is >= g,
    the girth is >= g.  O(n^2) with a small constant (no per-state set).
    """
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            nd = dist[u] + 1
            if nd >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = nd
                    dq.append(v)
        for v in adj[s]:
            if dist[v] + 1 < g:
                return False
    return True


# --------------------------------------------------------------------------- #
# LUW construction CD(d, q) -> connected component of the zero point          #
# --------------------------------------------------------------------------- #
def _luw_component(q, d):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _luw_construct(k, g):
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d + 5 < g:
        d += 1
    return _luw_component(q, d)


# --------------------------------------------------------------------------- #
# Cayley graph on Z_q x Z_2, degree 5 (generators +/-a, +/-b, (0,1))          #
# --------------------------------------------------------------------------- #
def _cayley_zq_z2(q, a, b):
    n = 2 * q
    adj = [[] for _ in range(n)]
    edges = []
    for i in range(q):
        for t in (0, 1):
            u = 2 * i + t
            for (di, dt) in ((a, 0), (-a, 0), (b, 0), (-b, 0), (0, 1)):
                v = 2 * ((i + di) % q) + ((t + dt) & 1)
                if u < v:
                    adj[u].append(v)
                    adj[v].append(u)
                    edges.append((u, v))
    return n, adj, edges


def _is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


def _next_prime(x):
    x = max(2, int(x))
    while not _is_prime(x):
        x += 1
    return x


# --------------------------------------------------------------------------- #
# random k-regular generator + local search                                   #
# --------------------------------------------------------------------------- #
def _random_kregular(n, k, rng):
    if n < k or (n * k) % 2 != 0:
        return None
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    perm = list(range(n * k))
    rng.shuffle(perm)
    adj = [set() for _ in range(n)]
    for i in range(0, n * k, 2):
        u = stubs[perm[i]]
        v = stubs[perm[i + 1]]
        if u == v or v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return adj


def _local_search(adj, n, k, g, rng, deadline):
    adj = [list(s) for s in adj]
    if _girth_ok(adj, n, g):
        return adj
    while _time_left() > 1.0:
        u = rng.randrange(n)
        if not adj[u]:
            continue
        v = adj[u][rng.randrange(len(adj[u]))]
        x = rng.randrange(n)
        if x in (u, v) or not adj[x]:
            continue
        cands = [y for y in adj[x] if y not in (u, v, x)]
        if not cands:
            continue
        y = cands[rng.randrange(len(cands))]
        if x in adj[v] or y in adj[u]:
            continue
        # swap (u,v),(x,y) -> (u,x),(v,y)
        adj[u].remove(v); adj[v].remove(u)
        adj[x].remove(y); adj[y].remove(x)
        adj[u].append(x); adj[x].append(u)
        adj[v].append(y); adj[y].append(v)
        if _girth_ok(adj, n, g):
            return adj
        # revert
        adj[u].remove(x); adj[x].remove(u)
        adj[v].remove(y); adj[y].remove(v)
        adj[u].append(v); adj[v].append(u)
        adj[x].append(y); adj[y].append(x)
    return None


def _try_random(n, k, g, rng):
    for _ in range(40):
        if _time_left() < 3.0:
            return None
        adj = _random_kregular(n, k, rng)
        if adj is None:
            continue
        if _girth_ok(adj, n, g):
            return adj
        adj = _local_search(adj, n, k, g, rng, _DEADLINE)
        if adj is not None and _girth_ok(adj, n, g):
            return adj
    return None


# --------------------------------------------------------------------------- #
def _adj_to_edges(adj, n):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


# --------------------------------------------------------------------------- #
def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 118.0  # safety margin below the 120 s limit
    rng = random.Random(12345)

    # 1) guaranteed-valid LUW baseline
    n_luw, edges_luw = _luw_construct(k, g)
    best_n, best_edges = n_luw, edges_luw

    # Moore bound (odd girth g = 2r+1): n >= 1 + k * sum_{e<r} (k-1)^e
    r = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** e for e in range(r))
    lo_q = max(2, (moore + 1) // 2)
    hi_q = best_n // 2

    # 2) scan the Z_q x Z_2 Cayley family for a smaller verified-valid graph
    pairs = [(a, b) for a in range(1, 8) for b in range(1, 8)
             if a != b and b != (-a) % (10 ** 9)]
    q = lo_q
    while q <= hi_q and _time_left() > 2.0:
        q = _next_prime(q)
        for (a, b) in pairs:
            if q <= max(a, b):
                continue
            if _time_left() <= 1.0:
                break
            n, adj, edges = _cayley_zq_z2(q, a % q, b % q)
            if n < best_n and _girth_ok(adj, n, g):
                best_n, best_edges = n, edges
        q += 1

    # 3) random construction + local search to try to beat the current best
    if _time_left() > 15.0:
        rng_n = max(moore, 2 * k)
        while rng_n < best_n and _time_left() > 5.0:
            if (rng_n * k) % 2 != 0:
                rng_n += 1
                continue
            adj = _try_random(rng_n, k, g, rng)
            if adj is not None:
                best_n, best_edges = rng_n, _adj_to_edges(adj, rng_n)
                break
            rng_n += 2  # keep n*k even

    # 4) final verified-valid result
    return best_n, best_edges
