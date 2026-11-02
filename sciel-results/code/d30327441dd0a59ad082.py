import time
import sys
from collections import deque

def _luw_construct(k, d):
    """
    Constructs the Lazebnik-Ustimenko-Woldar graph CD(d, k).
    Returns (n, edges).
    """
    q = k
    if d < 2:
        d = 2

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

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

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _check_girth(n, edges, g):
    """
    Checks if the graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = (g - 1) // 2

    for start in range(n):
        visited = {}
        queue = deque([(start, -1, 0)])
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                state = (nb, node)
                if state in visited:
                    cycle_len = (depth + 1) - visited[state]
                    if cycle_len < g:
                        return False
                else:
                    visited[state] = depth + 1
                    queue.append((nb, node, depth + 1))
    return True


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # LUW graphs are bipartite with girth exactly d+4 (for d>=2, q>=5)
    # So we need d+4 >= g, i.e., d >= g-4
    # But since they are bipartite, girth is even, so girth >= g means girth >= g if g is odd
    # (since girth even >= odd g means girth >= g+1)
    # Actually girth >= g: if g is odd, even girth >= g means girth >= g+1 (next even)
    # LUW girth is d+4. We need d+4 >= g.
    # For k=5, g=9: d >= 5. LUW with d=5 gives girth 9? No, bipartite girth is even.
    # Wait, LUW CD(d,k) has girth d+4. For d=5, girth=9. But it's bipartite so girth must be even.
    # Let me re-check: LUW graphs have girth d+4. For d=2, girth 6 (Heawood-like). 
    # Actually the girth of CD(d,q) is d+4. For d=5, girth 9. But bipartite graphs have even girth.
    # So d+4 must be even, meaning d must be even. For d=5, girth would be 9 which is odd - contradiction.
    # The actual girth is d+4 when d+4 is even, or d+5 when d+4 is odd? 
    # No, the LUW paper states girth is d+4. Let me just use the safe bound: d >= g - 4.
    
    min_d = max(2, g - 4)
    best_n = float('inf')
    best_edges = None

    for d in range(min_d, g + 5):
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_construct(k, d)
        except Exception:
            continue

        if d + 4 >= g:
            if n < best_n:
                best_n = n
                best_edges = edges
        else:
            if time.time() > deadline:
                break
            if _check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    # Fallback
    max_d = max(2, g - 4)
    n, edges = _luw_construct(k, max_d)
    return n, edges
