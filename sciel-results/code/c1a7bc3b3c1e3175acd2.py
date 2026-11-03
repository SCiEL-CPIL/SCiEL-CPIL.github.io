import time
import random
import itertools
from collections import deque

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


def _nonback_girth(adj, n):
    """Shortest cycle length, certified by non-backtracking walks from every
    vertex. A value >= g means no cycle shorter than g exists."""
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    while q:
        (v, p), (u, dist) = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                return dist + 1
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return 10 ** 9


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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


def _circulant_adj(n, S):
    adj = [set() for _ in range(n)]
    for i in range(n):
        for s in S:
            a, b = (i + s) % n, (i - s) % n
            if a != i and b != i:
                adj[i].add(a)
                adj[i].add(b)
    return adj


def _circulant(k, g, lo, hi):
    if k % 2 != 0:
        return None
    half = k // 2
    best = None
    n = lo
    while n <= hi and _elapsed() < _DEADLINE:
        for S in itertools.combinations(range(1, n // 2 + 1), half):
            if _elapsed() > _DEADLINE:
                break
            S = list(S)
            if any((n - s) % n in S for s in S):
                continue
            adj = _circulant_adj(n, S)
            if any(len(adj[i]) != k for i in range(n)):
                continue
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                best = (n, edges)
                break
        if best is not None:
            break
        n += 2
    return best


def construct(k, g):
    _start[0] = time.time()
    best = None

    luw = _luw(k, g)
    adj = _build_adj(luw[1], luw[0])
    if _is_valid(adj, luw[0], k, g):
        best = luw

    circ = _circulant(k, g, 2 * k, 2 * k + 120)
    if circ is not None:
        adj = _build_adj(circ[1], circ[0])
        if _is_valid(adj, circ[0], k, g):
            if best is None or circ[0] < best[0]:
                best = circ

    if best is None:
        best = luw
    return best[0], best[1]
