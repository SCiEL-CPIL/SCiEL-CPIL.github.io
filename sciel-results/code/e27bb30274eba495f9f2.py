"""Construct a k-regular graph of girth >= g on as few vertices as possible.

Searches random k-regular graphs at increasing vertex counts, certifying
girth >= g exactly (BFS from every vertex, equivalent to non-backtracking
walk certification).  A deadline guard returns the best valid graph found,
with an L-U-W explicit construction as a guaranteed-valid fallback.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound_lower(k, g):
    h = (g - 1) // 2
    s = 0
    for i in range(h + 1):
        s += k ** i
    return k * s


def _girth_at_least(n, adj, g):
    """Exact: True iff no cycle of length < g exists (BFS from every vertex)."""
    half = (g - 1) // 2
    for s in range(n):
        depth = [-1] * n
        parent = [-1] * n
        depth[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = depth[u]
            if du >= half:
                continue
            nd = du + 1
            for v in adj[u]:
                if depth[v] == -1:
                    depth[v] = nd
                    parent[v] = u
                    q.append(v)
        for u in range(n):
            du = depth[u]
            if du == -1:
                continue
            for v in adj[u]:
                dv = depth[v]
                if dv == -1:
                    continue
                if parent[u] == v or parent[v] == u:
                    continue
                L = du + dv + 1
                if L >= 3 and L < g:
                    return False
    return True


def _random_regular(n, k, rng, iters=1500):
    if n * k % 2 != 0 or k >= n:
        return None
    base = None
    for _ in range(300):
        stubs = [v for v in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        ok = True
        seen = set()
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                ok = False
                break
            key = (a, b) if a < b else (b, a)
            if key in seen:
                ok = False
                break
            seen.add(key)
            adj[a].append(b)
            adj[b].append(a)
        if ok:
            base = adj
            break
    if base is None:
        return None
    adj = [list(x) for x in base]
    for _ in range(iters):
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b or not adj[a] or not adj[b]:
            continue
        c = adj[a][rng.randrange(len(adj[a]))]
        d = adj[b][rng.randrange(len(adj[b]))]
        if c == b or d == a or c == d:
            continue
        if d in adj[a] or c in adj[b]:
            continue
        adj[a].remove(c); adj[c].remove(a)
        adj[b].remove(d); adj[d].remove(b)
        adj[a].append(d); adj[d].append(a)
        adj[b].append(c); adj[c].append(b)
    return adj


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    rng = random.Random(12345)

    lb = _moore_bound_lower(k, g)
    import math
    n_random_est = int(k ** ((g - 1) / 2.0)) + 1
    n_start = max(lb, n_random_est, 60)
    if n_start % 2 != 0:
        n_start += 1

    best_n, best_edges = None, None
    n = n_start
    max_n = 3000
    while n <= max_n and time.time() < deadline - 5.0:
        for trial in range(10):
            if time.time() > deadline - 3.0:
                break
            adj = _random_regular(n, k, rng, iters=1500)
            if adj is None:
                continue
            if not all(len(adj[v]) == k for v in range(n)):
                continue
            if _girth_at_least(n, adj, g):
                edges = [(v, w) for v in range(n) for w in adj[v] if v < w]
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
                return n, edges
        n += 2

    if best_n is not None:
        return best_n, best_edges
    if time.time() < deadline - 2.0:
        return _luw_construct(k, g)
    raise RuntimeError("No valid graph found within time budget")


def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

# EVOLVE-BLOCK-END
