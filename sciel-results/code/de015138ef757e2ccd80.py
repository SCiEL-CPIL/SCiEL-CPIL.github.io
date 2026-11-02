import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 115.0 # 5s margin
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
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

    def heuristic_girth_check(n, adj, g, sample_size=10):
        half_g = g // 2
        sources = random.sample(range(n), min(sample_size, n))
        for s in sources:
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

    def generate_simple_regular(n, k):
        # Configuration model with repair
        max_attempts = 30
        for attempt in range(max_attempts):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            edge_set = set()
            valid = True
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))
            
            if valid:
                return edges
                
        return None

    # Search
    best_n = seed_n
    best_edges = seed_edges
    
    # Try n values
    trials_per_n = 8
    
    for n in range(moore_bound, min(seed_n, moore_bound + 300)):
        if time.time() > deadline - 15:
            break
        
        for _ in range(trials_per_n):
            if time.time() > deadline - 15:
                break
            edges = generate_simple_regular(n, k)
            if edges is None:
                continue
            
            adj = build_adj(edges, n)
            
            # Quick heuristic check
            if heuristic_girth_check(n, adj, g, sample_size=5):
                # Full check
                if exact_girth_check(n, adj, g):
                    return n, edges
                    
    return best_n, best_edges
