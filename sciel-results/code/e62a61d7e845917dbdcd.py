import random
from collections import deque

def construct(k, g):
    # k=5, g=9
    def bfs_distance(u, v, adj, max_depth):
        if u == v:
            return 0
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= max_depth:
                continue
            for nei in adj[node]:
                if nei not in visited:
                    visited[nei] = dist + 1
                    if nei == v:
                        return dist + 1
                    queue.append((nei, dist + 1))
        return -1

    def is_connected(adj, n):
        if n == 0:
            return True
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for nei in adj[node]:
                if not visited[nei]:
                    visited[nei] = True
                    count += 1
                    queue.append(nei)
        return count == n

    def is_valid_edge(u, v, adj, degrees):
        if degrees[u] >= k or degrees[v] >= k:
            return False
        if v in adj[u]:
            return False
        d = bfs_distance(u, v, adj, g - 2)
        if d == -1:
            return True
        if d >= g - 1:
            return True
        return False

    start_n = 426
    if start_n % 2 != 0:
        start_n += 1
        
    max_n = 1250
    
    for n in range(start_n, max_n + 1, 2):
        for attempt in range(10):
            random.seed(n * 100 + attempt)
            adj = [[] for _ in range(n)]
            degrees = [0] * n
            
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))
            random.shuffle(edges)
            
            for u, v in edges:
                if is_valid_edge(u, v, adj, degrees):
                    adj[u].append(v)
                    adj[v].append(u)
                    degrees[u] += 1
                    degrees[v] += 1
            
            if all(d == k for d in degrees) and is_connected(adj, n):
                final_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            final_edges.append((i, j))
                return n, final_edges

    # Fallback to LUW
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
