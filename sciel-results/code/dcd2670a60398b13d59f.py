import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    def luw_construction():
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

    def has_short_cycle(adj, g):
        n = len(adj)
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                v = q.popleft()
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        q.append(w)
                    else:
                        # Check if this edge creates a cycle
                        # Cycle length = dist[v] + dist[w] + 1
                        # But this is only valid if the paths are disjoint?
                        # Actually, for BFS, if we see a visited node, the cycle length is dist[v] + dist[w] + 1.
                        # However, this might count the same cycle multiple times or incorrectly.
                        # A safer check: 
                        # If dist[v] + dist[w] + 1 < g, return True.
                        if dist[v] + dist[w] + 1 < g:
                            return True
        return False

    def try_local_search(N, k, g, deadline):
        # Generate random k-regular graph
        # ...
        pass

    N_target = 320
    # Implement try_local_search properly
    # ...
    
    # Fallback
    return luw_construction()
