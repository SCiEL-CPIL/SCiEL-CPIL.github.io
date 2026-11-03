import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW construction
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

    # Random Graph Generator
    def generate_random_3_regular(N):
        # Configuration model
        while True:
            stubs = []
            for i in range(N):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            edges = []
            seen = set()
            valid = True
            for i in range(0, 2 * N * 3 // 2, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in seen:
                    valid = False
                    break
                seen.add((u, v))
                edges.append((u, v))
            if valid:
                return edges

    # Girth Checker
    def has_short_cycle(edges, N, min_girth):
        adj = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        # The cycle length is dist[u] + 1 + dist[v] ?
                        # Only if u and v are not parent-child in the tree
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + 1 + dist[v]
                            if cycle_len < min_girth:
                                return True
        return False

    best_n = float('inf')
    best_edges = None
    
    # Try to find a graph with N close to the Moore bound
    # Moore bound for k=3, g=16 is 510.
    # Known best is 936.
    # We try N from 936 upwards.
    
    # To save time, we can jump.
    # If 936 fails, try 1000, 1200, 1500, 2000.
    # If 2000 succeeds, we can try to go down again?
    # No, just find the first valid one.
    
    # Heuristic: Random graphs have girth ~ 2 log_2 N.
    # For girth 16, we need 2 log_2 N >= 16 => log_2 N >= 8 => N >= 256.
    # But expected number of cycles grows.
    # For N=936, E[C_3] ~ 1, E[C_4] ~ 2.
    # So it's unlikely to be girth 16.
    # For N=2000, E[C_3] ~ 1, E[C_15] ~ 1000?
    # Wait, E[C_k] ~ 2^{k-1}/k.
    # For k=15, E[C_15] ~ 1000.
    # So even for N=2000, there are likely short cycles.
    # The girth of a random 3-regular graph is small!
    # My previous analysis was correct.
    # So random search is NOT a good strategy for high girth.
    
    # We MUST use the LUW construction or a similar explicit high-girth construction.
    # The LUW construction is the only one I know how to implement quickly.
    # It gives 177147.
    
    # Is there a way to improve LUW?
    # No.
    
    # So I will return the LUW graph.
    # But I will add a timeout guard.
    
    return luw_construct(k, g)
