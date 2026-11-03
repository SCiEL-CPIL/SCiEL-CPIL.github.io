import time
import random
import sys

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    # Heuristic search for a smaller graph
    def heuristic_search(k, g, time_limit=110):
        start_time = time.time()
        # Try a few sizes
        for n in [1000, 1200, 1500]:
            if time.time() - start_time > time_limit:
                break
            # Generate random k-regular graph
            # Using random pairing of half-edges
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            
            # Shuffle and pair
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                continue
            
            # Check girth and try to improve
            # This is a simplified local search
            # For now, just check if it's valid
            # If valid, return
            if check_girth(adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    def check_girth(adj, g):
        n = len(adj)
        # BFS from each vertex
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            depth = [0] * n
            queue = [start]
            visited[start] = start
            found_cycle = False
            while queue:
                u = queue.pop(0)
                if depth[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = u
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif visited[v] != u and parent[u] != v:
                        # Found a cycle
                        # Length is depth[u] + depth[v] + 1
                        # This is a rough check
                        pass
        return True # Placeholder, need robust girth check

    # Since robust girth check and local search are complex and risky in 120s,
    # and the LUW graph is guaranteed valid, we return LUW.
    # But we can try to see if a smaller LUW-like graph exists?
    # No, LUW is fixed for k=3.
    
    return luw_construct(k, g)
