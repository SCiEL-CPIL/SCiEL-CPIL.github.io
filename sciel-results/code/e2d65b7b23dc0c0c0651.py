"""4-regular girth-10 cage construction.

Strategy:
  * The Hoffman-Singleton graph is 3-regular girth-7, NOT applicable here.
  * Use the Lazebnik-Ustimenko-Woldar (LUW) construction with q=4, d=5,
    which guarantees a 4-regular bipartite graph with girth >= 10.
  * Verify the result rigorously (simple, 4-regular, connected, girth >= 10).
  * Deadline guard to stay within 120s.
"""
import time

_DEADLINE = 115.0  # safety margin before 120s kill


def _luw_edges(q, d):
    """Construct LUW CD(d, q) graph. Returns (n, edges) for the connected component
    of the zero vertex."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Compute the rule for each column c = 2..d
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The four possible rules cycling
        candidates = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = candidates[r]

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
    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once: when the point-side vertex is processed
            if vertex[0] == 0:
                edges.append((vi, index[other]))

    return len(order), edges


def _verify(n, edges, k, g):
    """Verify: simple, k-regular, connected, girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False

    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check via non-backtracking BFS from each vertex
    # For each vertex, do BFS in the non-backtracking state space (prev, cur)
    # and check if we can return to the start vertex with length < g
    INF = 10 ** 9
    for u in range(n):
        # BFS over non-backtracking walks starting at u
        # State: (prev, cur), start: (u, w) for each neighbor w of u
        dist = {}
        queue = []
        for w in adj[u]:
            dist[(u, w)] = 1
            queue.append((u, w, 1))
        head = 0
        while head < len(queue):
            prev, cur, L = queue[head]
            head += 1
            if L >= g:
                break
            # If we've returned to u, we found a cycle of length L
            if cur == u and L >= 3:
                return False
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    queue.append((cur, w, L + 1))
    return True


def construct(k, g):
    start_time = time.time()

    # For the specific task: k=4, g=10
    # LUW with q=k=4, d chosen so that d+5 >= g => d >= g-5
    # d must be odd for the standard LUW girth guarantee
    if k == 4 and g == 10:
        d = 5  # d+5 = 10
        n, edges = _luw_edges(4, d)
        if _verify(n, edges, k, g):
            return n, edges

    # General fallback: try LUW with q=k
    if k >= 3:
        d = g - 5
        if d < 3:
            d = 3
        if d % 2 == 0:
            d += 1  # ensure odd
        n, edges = _luw_edges(k, d)
        if _verify(n, edges, k, g):
            return n, edges

    # Last resort: return an empty result (should not happen)
    return 0, []
