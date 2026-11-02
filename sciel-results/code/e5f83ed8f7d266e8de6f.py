import time
import random
import numpy as np
from collections import deque

def _seed_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def _generate_random_k_regular(n, k):
    stubs = list(range(n * k))
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, n * k, 2):
        u = stubs[i] // k
        v = stubs[i+1] // k
        if u == v:
            return None
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            return None
        edges.add((u, v))
        
    return list(edges)

def _has_girth_at_least(edges, n, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            u, prev, dist = queue.popleft()
            
            if dist == g - 1:
                continue
                
            for v in adj[u]:
                if v == prev:
                    continue
                    
                if v == start and dist + 1 >= 2:
                    return False
                
                state = (v, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, u, dist + 1))
                    
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a better graph
    # Moore bound
    moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    # We search for n starting from moore + 10
    # Step size 20
    for n in range(max(50, moore + 10), 2000, 20):
        if time.time() > deadline:
            break
            
        for attempt in range(30):
            if time.time() > deadline:
                break
                
            edges = _generate_random_k_regular(n, k)
            if edges is None:
                continue
                
            if _has_girth_at_least(edges, n, g):
                return n, edges
                
    # Fallback
    return _seed_construct(k, g)
