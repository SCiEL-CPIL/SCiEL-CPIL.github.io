import random
import time
from collections import deque
import numpy as np

def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    for n in range(766, 1000, 2):
        if n % 2 != 0:
            continue
        half = n // 2
        # Try to build a random 3-regular bipartite graph
        # Increase attempts
        for attempt in range(1000):
            if time.time() - start_time > time_limit:
                break
            edges = _random_3reg_bipartite(half)
            if edges is None:
                continue
            if _check_girth_fast(edges, n, g):
                return n, edges
        
        if time.time() - start_time > time_limit:
            break
            
    return _luw_construct(k, g)

def _random_3reg_bipartite(half):
    """Generate a random 3-regular bipartite graph with half vertices on each side."""
    n = half * 2
    
    stubs_a = [i for i in range(half) for _ in range(3)]
    stubs_b = [i + half for i in range(half) for _ in range(3)]
    
    random.shuffle(stubs_a)
    random.shuffle(stubs_b)
    
    edges = set()
    for a, b in zip(stubs_a, stubs_b):
        if a > b:
            a, b = b, a
        if (a, b) in edges:
            return None
        edges.add((a, b))
        
    # Quick degree check
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
        
    return list(edges)

def _check_girth_fast(edges, n, g):
    """Check if graph has girth at least g using array-based BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # We need to check for cycles of length < g
    # In bipartite graph, min cycle length is even.
    # Max depth to check: (g-2)//2
    max_depth = (g - 2) // 2
    
    dist = [-1] * n
    parent = [-1] * n
    
    for start in range(n):
        # Reset dist and parent? 
        # Doing full reset is O(n^2). 
        # Better to use a "visited" marker with timestamps or just re-allocate.
        # Given n~1000, re-allocation is fine.
        dist[:] = [-1] * n
        parent[:] = [-1] * n
        
        dist[start] = 0
        queue = deque([(start, 0)])
        
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            for neighbor in adj[node]:
                if neighbor == parent[node]:
                    continue
                if dist[neighbor] != -1:
                    # Found a cycle
                    cycle_len = dist[neighbor] + depth + 1
                    if cycle_len < g:
                        return False
                else:
                    dist[neighbor] = depth + 1
                    parent[neighbor] = node
                    queue.append((neighbor, depth + 1))
                    
    return True

def _luw_construct(k, g):
    # Keep the same LUW construction as before, it's the fallback
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
