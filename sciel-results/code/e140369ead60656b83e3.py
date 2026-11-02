import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
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

    # Heuristic Search for smaller graph
    def check_girth(n, edges):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = float('inf')
        # BFS from each vertex
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = [i]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            # Optimization: if min_girth <= g, we can stop early? 
            # But we need to know if it's >= g.
            # If we find a cycle < g, we can return early.
            if min_girth < g:
                return min_girth
        return min_girth

    def random_3_regular(n):
        # Generate a random 3-regular graph
        # Use configuration model
        if n < 2 or n % 2 != 0:
            return None
        half_edges = [i for i in range(n) for _ in range(3)]
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        used = [False] * len(half_edges)
        edges = []
        for i in range(len(half_edges)):
            if used[i]:
                continue
            u = half_edges[i]
            # Find a partner
            found = False
            for j in range(i+1, len(half_edges)):
                if used[j]:
                    continue
                v = half_edges[j]
                if u == v:
                    continue
                if v in adj[u]:
                    continue
                # Add edge
                adj[u].append(v)
                adj[v].append(u)
                used[i] = True
                used[j] = True
                edges.append((u, v))
                found = True
                break
            if not found:
                return None
        # Check if all degrees are 3
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return edges

    # Try to find a small graph
    # Start with small N, increase if not found
    # Note: 512 is the lower bound.
    for N in range(512, 2000, 2):
        if time.time() > deadline:
            break
        # Try a few random instances
        for _ in range(50):
            if time.time() > deadline:
                break
            edges = random_3_regular(N)
            if edges is None:
                continue
            girth = check_girth(N, edges)
            if girth >= g:
                return N, edges
    
    # If no small graph found, use LUW
    return luw_construct(k, g)
