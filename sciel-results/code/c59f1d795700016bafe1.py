import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    def seed_construct(k, g):
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

    seed_n, seed_edges = seed_construct(k, g)
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def voltage_graph_construct(k, g):
        """Construct a k-regular graph with girth >= g using a voltage graph over a group.
        
        For k=5, g=9: Use the group SL(2,11) with carefully chosen generators.
        The Cayley graph of a group with a symmetric generating set S (|S|=k) is
        k-regular. We need to verify girth >= g by checking no short relations.
        
        Alternative: Use a smaller group or a different approach.
        For now, try a known construction or fall back to seed.
        """
        if k != 5 or g != 9:
            return None
        
        # For (5,9), try to use a voltage graph construction.
        # One approach: Use the group of order 1320 (SL(2,11) has order 1320).
        # But we need to find 5 generators such that no word of length < 9 is identity.
        
        # Alternative: Use a smaller group. The Moore bound is 1 + 5*4^4 = 1 + 5*256 = 1281.
        # Actually for girth 9 (odd), the Moore bound is 1 + k * sum_{i=0}^{r-1} (k-1)^i where r=(g-1)/2=4
        # = 1 + 5 * (1 + 4 + 16 + 64) = 1 + 5 * 85 = 1 + 425 = 426.
        # Wait, let me recalculate. For odd girth g=2r+1, the Moore bound is:
        # 1 + k * sum_{i=0}^{r-1} (k-1)^i = 1 + k * ((k-1)^r - 1)/(k-2) for k>2
        # For k=5, r=4: 1 + 5 * (4^4 - 1)/(5-2) = 1 + 5 * (256-1)/3 = 1 + 5*85 = 426.
        
        # The seed construction gives a graph. Let's try to improve it.
        # Actually, let's try a different approach: use a known small (5,9)-cage or close to it.
        
        # For now, return None to fall back to seed.
        # TODO: Implement a proper voltage graph construction.
        return None
    
    # Try voltage graph construction
    result = voltage_graph_construct(k, g)
    if result is not None:
        n, edges = result
        adj = build_adj(edges, n)
        # Verify
        if exact_girth_check(n, adj, g):
            return n, edges
    
    # Fallback to seed
    return seed_n, seed_edges
