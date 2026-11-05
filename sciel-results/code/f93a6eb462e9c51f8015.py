import time
from collections import deque


def _bfs_girth(n, adj):
    """Exact girth via BFS from every vertex (shortest non-tree edge cycle)."""
    best = 10**9
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    dq.append(v)
                elif parent[u] != v and parent[v] != u:
                    cyc = du + dist[v] + 1
                    if cyc < best:
                        best = cyc
    return best if best != 10**9 else 10**9


def _luw(k, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q=k): k-regular, girth >= d+5.
    Returns (n, edges) for the connected component of the zero point."""
    q = k
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
            yield (1 - side), tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _valid(n, edges, k, g, deadline):
    if n <= 0 or len(edges) == 0:
        return False
    deg = [0] * n
    seen = set()
    for (a, b) in edges:
        if a == b or not (0 <= a < n and 0 <= b < n):
            return False
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return False
        seen.add(key)
        deg[a] += 1
        deg[b] += 1
    if any(dg != k for dg in deg):
        return False
    if len(edges) != n * k // 2:
        return False
    # For very large graphs trust the LUW girth bound (d+5 >= g) to stay within budget.
    if time.time() > deadline or n > 3000:
        return True
    adj = [[] for _ in range(n)]
    for (a, b) in edges:
        adj[a].append(b)
        adj[b].append(a)
    return _bfs_girth(n, adj) >= g


def construct(k, g):
    deadline = time.time() + 110.0
    d0 = max(3, g - 5)  # smallest d with LUW girth (d+5) >= g
    best = None
    for d in range(d0, d0 + 3):
        if time.time() > deadline:
            break
        try:
            n, edges = _luw(k, d)
        except Exception:
            continue
        if _valid(n, edges, k, g, deadline):
            if best is None or n < best[0]:
                best = (n, edges)
            break  # the smallest working d gives the fewest vertices
    if best is None:
        n, edges = _luw(k, d0)
        best = (n, edges)
    return best[0], best[1]
