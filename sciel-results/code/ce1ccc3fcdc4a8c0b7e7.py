import time
import random
from collections import deque

def _verify_girth_circulant(n, steps, g):
    """
    Verify girth of a circulant graph C(n, steps) by BFS from 0.
    Since circulants are vertex-transitive, girth from 0 is the global girth.
    """
    # Build adjacency for BFS
    # steps is a list of 5 steps.
    adj = []
    for u in range(n):
        neighbors = [(u + s) % n for s in steps]
        adj.append(neighbors)
        
    # BFS from 0
    # dist[v] = shortest distance from 0
    # parent[v] = parent in BFS tree
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    q = deque([0])
    
    while q:
        u = q.popleft()
        if dist[u] >= g - 1:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
            elif v != parent[u]:
                # Found a cycle
                # Cycle length = dist[u] + dist[v] + 1
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return False
    return True

def _construct_luw(k, g):
    """Fallback LUW construction."""
    q = k
    d_min = max(3, (g - 2 + 1) // 2)
    if d_min % 2 == 0:
        d_min += 1
    d = d_min
    
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
    n = len(order)
    return n, edges

def construct(k, g):
    deadline = time.time() + 115
    
    # Moore bound
    if g % 2 == 0:
        r = g // 2 - 1
        n_min = 2 * sum((k-1)**i for i in range(r+1))
    else:
        r = (g - 1) // 2
        n_min = 1 + k * sum((k-1)**i for i in range(r))
        
    if k == 5:
        # Strategy: Search for a circulant graph
        # n must be even.
        # Steps: {n/2, a, n-a, b, n-b}
        
        # Define search ranges
        # 1. Small n (just in case)
        # 2. Around known optimum 1116
        
        ranges = []
        start_small = n_min
        if start_small % 2 != 0:
            start_small += 1
        ranges.append((start_small, start_small + 100))
        
        # Target range around 1116
        ranges.append((1100, 1150))
        
        # Also try a bit higher
        ranges.append((1150, 1200))
        
        for start, end in ranges:
            if start % 2 != 0:
                start += 1
            for n in range(start, end + 1, 2):
                if time.time() > deadline - 10:
                    break
                
                # Try random step sets
                # We need 2 pairs from {1, ..., n/2 - 1}
                # To avoid girth 4, we should avoid a+b = n or a+b=0 etc?
                # Actually, just check with BFS.
                
                attempts = 0
                max_attempts = 20
                while attempts < max_attempts:
                    attempts += 1
                    if time.time() > deadline - 5:
                        break
                    
                    candidates = list(range(1, n//2))
                    if len(candidates) < 2:
                        continue
                    a, b = random.sample(candidates, 2)
                    steps = [n//2, a, n-a, b, n-b]
                    
                    # Quick check: No duplicate steps
                    if len(set(steps)) != 5:
                        continue
                    
                    if _verify_girth_circulant(n, steps, g):
                        # Construct edges
                        edges = []
                        for u in range(n):
                            for s in steps:
                                v = (u + s) % n
                                if u < v:
                                    edges.append((u, v))
                        return n, edges
                        
    # Fallback
    return _construct_luw(k, g)
