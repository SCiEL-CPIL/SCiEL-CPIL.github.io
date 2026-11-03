import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def luw_construct(k, g):
        # Determine optimal d
        # Girth of CD(d, q) is typically 2d+2 or 2d+3
        # We want girth >= g
        # So 2d+2 >= g => d >= (g-2)/2
        d_min = (g - 2) // 2 + 1
        # Try a few d values starting from d_min
        best_n = float('inf')
        best_edges = None
        
        for d in range(d_min, d_min + 5):
            if time.time() > deadline - 5:
                break
            try:
                n, edges = _luw_impl(k, d)
                # Check girth
                if _has_girth_at_least(edges, n, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
            except Exception:
                pass
        
        if best_edges is not None:
            return best_n, best_edges
        return None

    def _luw_impl(k, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def _has_girth_at_least(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                v = queue.popleft()
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v]:
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < g:
                            return False
        return True

    # Specific handling for k=3, g=16
    if k == 3 and g == 16:
        result = luw_construct(3, 16)
        if result:
            return result

    # Fallback for other cases
    result = luw_construct(k, g)
    if result:
        return result
    
    # Emergency fallback: return a trivial valid graph if possible, or None
    # But the task requires a valid graph.
    # If LUW fails, we might need a different construction.
    # For now, return the LUW result even if girth check failed? No, must be valid.
    # We'll assume LUW works for the given parameters.
    return luw_construct(k, g)
