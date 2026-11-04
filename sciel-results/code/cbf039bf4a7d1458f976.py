"""Construct a small 4-regular graph of girth >= 10 (cage n(4,10) upper bound)."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

_DEADLINE = None
_TIME_BUDGET = 112.0  # seconds, safety margin under 120s wall-clock


def _remaining():
    if _DEADLINE is None:
        return _TIME_BUDGET
    return max(0.0, _DEADLINE - time.time())


def _is_4regular(n, edges):
    deg = [0] * n
    seen = set()
    for (u, v) in edges:
        if u == v:
            return False
        if not (0 <= u < n and 0 <= v < n):
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != 4:
            return False
    return True


def _girth_ok(n, edges, g):
    """True iff no cycle of length < g. Non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    for s in range(n):
        dist = {}
        dq = deque()
        for w in adj[s]:
            st = (s, w)
            dist[st] = 1
            dq.append(st)
        while dq:
            prev, cur = dq.popleft()
            L = dist[(prev, cur)]
            if L >= g:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s:
                    if L + 1 < g:
                        return False
                    continue
                st = (cur, nxt)
                if st in dist:
                    continue
                dist[st] = L + 1
                dq.append(st)
    return True


def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_BUDGET

    best_n = None
    best_edges = None

    def try_candidate(n, edges):
        nonlocal best_n, best_edges
        if _remaining() <= 2.0:
            return
        if best_n is not None and n >= best_n:
            return
        if len(edges) != 2 * n:  # 4-regular simple => 2n edges
            return
        if not _is_4regular(n, edges):
            return
        if not _girth_ok(n, edges, g):
            return
        best_n = n
        best_edges = edges

    # Circulant 4-regular graphs C(n; +/−s1, +/−s2).
    # n must be even-ish for symmetry; search n in [244, ~400], sample step pairs.
    n_lo = 244
    n_hi = min(400, n_lo + int(_remaining() * 25))
    for n in range(n_lo, n_hi + 1):
        if _remaining() <= 6.0:
            break
        rng = random.Random(12345 + n)
        # candidate steps: 1 <= s1 < s2 < n/2, avoid s2 == 2*s1 (creates 4-cycles) etc.
        cands = []
        for s1 in range(1, n // 2):
            for s2 in range(s1 + 1, n // 2):
                if s2 == 2 * s1:
                    continue
                cands.append((s1, s2))
        if len(cands) > 150:
            cands = rng.sample(cands, 150)
        for (s1, s2) in cands:
            if _remaining() <= 6.0:
                break
            edges = []
            seen = set()
            ok = True
            for v in range(n):
                for s in (s1, -s1, s2, -s2):
                    w = (v + s) % n
                    if w == v:
                        ok = False
                        break
                    a, b = (v, w) if v < w else (w, v)
                    if (a, b) in seen:
                        ok = False
                        break
                    seen.add((a, b))
                    edges.append((a, b))
                if not ok:
                    break
            if ok:
                try_candidate(n, edges)

    if best_edges is not None:
        return best_n, best_edges

    return _seed_construct(k, g)


def _seed_construct(k, g):
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
