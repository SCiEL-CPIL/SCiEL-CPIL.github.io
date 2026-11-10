import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120.0

    # Try to construct the 320-vertex (4,10)-cage
    if k == 4 and g == 10:
        n = 320
        edges = []
        # Left vertices 0-159, Right vertices 160-319
        for u in range(160):
            i = u % 16
            j = u // 16
            # Neighbor 1: same i, j+1
            v1 = 160 + (i * 10 + (j + 1) % 10)
            # Neighbor 2: same i, j-1
            v2 = 160 + (i * 10 + (j - 1) % 10)
            # Neighbor 3: i+1, j+2i
            v3 = 160 + (((i + 1) % 16) * 10 + (j + 2 * i) % 10)
            # Neighbor 4: i-1, j-2i
            v4 = 160 + (((i - 1) % 16) * 10 + (j - 2 * i) % 10)
            
            edges.append((u, v1))
            edges.append((u, v2))
            edges.append((u, v3))
            edges.append((u, v4))
        
        # Verify girth
        if _check_girth(n, edges, g):
            return n, edges

    # Fallback: Lazebnik-Ustimenko-Woldar graph CD(d, q)
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

def _check_girth(n, edges, min_girth):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles shorter than min_girth using BFS from each vertex
    # This is O(n * (n+k)) which is fine for n=320
    for start in range(n):
        visited = [-1] * n
        prev = [-1] * n
        dist = [0] * n
        queue = [start]
        visited[start] = 0
        dist[start] = 0
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= min_girth // 2:
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = 1
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True
