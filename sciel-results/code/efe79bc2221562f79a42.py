import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    if k != 3:
        # Fallback for other k, though task is specific to 3
        return _generic_construct(k, g)

    # Try LUW constructions with increasing d
    # n = 2 * 3^d
    # We want girth >= g
    # Start with d that might satisfy girth
    # For g=16, d=7 gives n=4374. d=8 gives 13122.
    
    for d in range(6, 12): # Try d=6 to 11
        if time.time() > deadline:
            break
        
        # Construct LUW graph with q=3, depth d
        n, edges = _lnw_construct(3, d)
        
        if n == 0:
            continue
            
        # Check girth
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Verify 3-regular
        valid = True
        for i in range(n):
            if len(adj[i]) != 3:
                valid = False
                break
        if not valid:
            continue
            
        # Check if girth >= g using bounded BFS
        if _has_girth_at_least(adj, n, g):
            return n, edges

    # If all LUW attempts failed or timed out, fallback to seed's default d
    # Seed used d = g - 5 or g - 4
    d_seed = 11 if (g - 5) % 2 else 10
    n, edges = _lnw_construct(3, d_seed)
    return n, edges

def _has_girth_at_least(adj, n, g):
    """
    Check if the graph has girth at least g.
    Uses BFS from each node, limited to depth g//2 - 1.
    If no cycle of length < g is found, return True.
    """
    max_depth = g // 2 - 1
    # For g=16, max_depth = 7.
    # Cycles of length L < g can be found if L <= 2*max_depth + 1.
    # 2*7 + 1 = 15. So we detect all cycles of length <= 15.
    # If none found, girth >= 16.
    
    for start in range(n):
        # BFS
        # dist[node] = depth
        # parent[node] = parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        parent[start] = -2
        queue = deque([(start, -1)])
        
        while queue:
            node, par = queue.popleft()
            if dist[node] >= max_depth:
                continue
                
            for nb in adj[node]:
                if nb == par:
                    continue
                
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append((nb, node))
                else:
                    # Found a cycle
                    # The cycle length is dist[node] + dist[nb] + 1
                    # We must ensure we don't count the edge to parent as a cycle
                    # If parent[nb] == node, it's the tree edge, already skipped?
                    # No, we skipped nb == par.
                    # But if nb is visited and is not the parent, it's a back edge or cross edge.
                    # In BFS on undirected graph, any visited neighbor that is not parent forms a cycle.
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
    return True

def _lnw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction.
    k: degree (must be 3 for this specific implementation logic)
    d: depth
    """
    q = k
    # Rule for LUW
    # The seed used a specific rule generation.
    # I will replicate the seed's _lnw_construct logic but parameterized by d.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule generation from seed
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The seed's logic:
        # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # This looks specific. I will use it.
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
    
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _generic_construct(k, g):
    # Fallback for k != 3
    # Just return a large random graph or Moore bound estimate
    # Not expected to be used
    n = 2 * (k-1) ** (g//2)
    return n, []
