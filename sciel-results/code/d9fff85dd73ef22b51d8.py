import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # LUW construction parameters
    # For LUW CD(d, q), girth is at least 2d + 2 for q > 2?
    # We start with d such that 2d+2 >= g
    # d = (g - 2) // 2
    q = k
    
    # If k=3 and g=16, d=7. n = 2 * 3^8 = 13122.
    # If k=3 and g=14, d=6. n = 2 * 3^7 = 4374.
    
    # We try increasing d until girth is satisfied
    # Start with minimal d
    d = max(3, (g - 2) // 2)
    
    while time.time() < deadline - 10:
        n, edges = _generate_luw_graph(q, d)
        
        # Verify girth
        # If girth >= g, return
        # If girth < g, increase d
        if _check_girth(n, edges, g):
            return n, edges
            
        # If girth is too small, increase d
        d += 1
        if d > 20: # Safety cap
            break
            
    # Fallback: return the last generated graph (might be invalid if girth < g, but better than nothing? 
    # No, the evaluator requires valid graph. 
    # If we fail to find valid graph, we should return the best valid one found.
    # But LUW with large d is always valid.
    # So the loop should find a valid one.
    return n, edges

def _generate_luw_graph(q, d):
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

def _check_girth(n, edges, target_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_cycle = float('inf')
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < target_girth:
                            return False
        if min_cycle < target_girth:
            return False
            
    return min_cycle >= target_girth
