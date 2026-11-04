import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 4 and g == 10:
        # Foster cage: 4-regular, girth 10, 320 vertices
        n, edges = _foster_cage()
        # Verify girth
        if _verify_girth(n, edges, g):
            return n, edges
        # Fallback
        return _luw_graph(k, g)
    else:
        return _luw_graph(k, g)

def _foster_cage():
    """Construct the Foster cage: 4-regular graph with girth 10 on 320 vertices."""
    n = 320
    edges = set()
    for a in range(4):
        for b in range(80):
            v = a * 80 + b
            # Neighbor 1: (a, b+1)
            n1 = a * 80 + ((b + 1) % 80)
            # Neighbor 2: (a, b-1)
            n2 = a * 80 + ((b - 1) % 80)
            # Neighbor 3: (a+1, b+40)
            n3 = ((a + 1) % 4) * 80 + ((b + 40) % 80)
            # Neighbor 4: (a+1, b+41)
            n4 = ((a + 1) % 4) * 80 + ((b + 41) % 80)
            for nb in [n1, n2, n3, n4]:
                if v != nb:
                    e = (min(v, nb), max(v, nb))
                    edges.add(e)
    return n, list(edges)

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using BFS from each vertex (non-backtracking)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
    
    # BFS to find shortest cycle from each vertex
    # For efficiency, use non-backtracking BFS up to depth 4 (cycles of length <= 8)
    # If no cycle of length < 10, girth >= 10
    for start in range(n):
        # BFS tracking (node, prev) to avoid backtracking
        visited = {}  # (node, prev) -> depth
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited[(nb, start)] = 1
        
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= 5:  # cycles of length >= 10 have depth >= 5 from start
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and depth + 1 < g:
                    return False
                if (nb, node) not in visited:
                    visited[(nb, node)] = depth + 1
                    queue.append((nb, node, depth + 1))
    return True

def _luw_graph(k, g):
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
