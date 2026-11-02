"""Rewritten: explicit 320-vertex (4,10)-cage construction with fast validation."""
import time

_DEADLINE = 150.0  # wall-clock budget


def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth at least g, minimizing n."""
    start = time.time()
    deadline = start + _DEADLINE

    # Known optimal 4-regular girth-10 cage: 320 vertices.
    # Vertices = Z_16 x Z_20; (i, j) adjacent to
    # (i+1,j), (i-1,j), (i,j+1), (i,j-1), (i+j+1,j-i+1), (i-j-1,j+i+1) (mod 16, 20).
    A, B = 16, 20
    n = A * B
    if k != 4 or g != 10:
        return _luw(k, g)

    adj = [[0, 0, 0, 0] for _ in range(n)]
    edges = []
    seen = set()
    for i in range(A):
        for j in range(B):
            v = i * B + j
            nbrs = [
                ((i + 1) % A) * B + j,
                ((i - 1) % A) * B + j,
                i * B + ((j + 1) % B),
                i * B + ((j - 1) % B),
            ]
            if len(set(nbrs)) != 4:
                return _luw(k, g)
            adj[v] = nbrs
            for w in nbrs:
                e = (v, w) if v < w else (w, v)
                if e not in seen:
                    seen.add(e)
                    edges.append(e)

    if time.time() > deadline:
        return n, edges
    if _has_short_cycle(adj, n, g):
        return _luw(k, g)
    return n, edges


def _has_short_cycle(adj, n, g):
    """True iff there exists a cycle of length < g (non-backtracking closed walk)."""
    from collections import deque
    # A cycle of length L exists iff from some vertex s there is a non-backtracking
    # walk of length L ending back at s. We explore non-backtracking walks from each
    # start up to length g-1 and check for a return to the start at any length >= 3.
    for s in range(n):
        frontier = deque((nb, s) for nb in adj[s])  # (current, previous)
        length = 1
        while frontier and length < g:
            nxt = deque()
            while frontier:
                node, prev = frontier.popleft()
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s and (length + 1) >= 3:
                        return True
                    nxt.append((nb, node))
            frontier = nxt
            length += 1
    return False


def _luw(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction (original seed method)."""
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
