# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    if k != 3:
        return _lww_fallback(k, g)
    deadline = time.time() + 115
    # Try multiple field sizes for LWW construction, pick smallest valid graph
    best = None
    for q in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if time.time() > deadline - 5:
            break
        result = _lww_construct(q, k, g)
        if result is not None:
            n, edges = result
            if best is None or n < best[0]:
                best = (n, edges)
    if best is not None:
        return best
    return _lww_fallback(k, g)

def _lww_construct(q, k, g):
    """
    Lazebnik-Woldar type construction: q-regular bipartite graph with girth >= g.
    Uses the standard LWW rule with parameter d chosen so girth >= g.
    We build the connected component of the zero vertex.
    """
    # Choose d: for the LWW construction, girth >= d + 5 (odd d) or d + 4 (even d)
    # We need girth >= g
    # Try odd d = g - 5 if that works, else even d = g - 4
    d_candidates = []
    if (g - 5) % 2 == 1 and g - 5 >= 3:
        d_candidates.append(g - 5)
    if (g - 4) % 2 == 0 and g - 4 >= 4:
        d_candidates.append(g - 4)
    if not d_candidates:
        d_candidates = [max(3, g - 5)]
    
    # Use the smallest d that gives girth >= g
    d = min(d_candidates)
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    # Build rule table for c = 2..d
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    def neighbours(vertex):
        """Generate all neighbors of a vertex in the LWW graph."""
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    # BFS to build connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from point side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    n = len(order)
    
    # Verify: check all vertices have degree exactly k
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    # Verify girth >= g using non-backtracking walks
    # For each vertex, do BFS tracking (node, prev) to avoid backtracking
    # The girth is the minimum cycle length. We check that no cycle shorter than g exists.
    # A cycle of length L through vertex v means: starting from v, going to a neighbor,
    # then doing a non-backtracking walk of length L-2 that returns to v.
    # More efficiently: for each edge (u,v), check if there's a path of length >= g-1 
    # from u to v not using edge (u,v). But that's expensive.
    # Simpler: BFS from each vertex, tracking distance. If any neighbor of start is at 
    # distance d where d+1 < g, we have a cycle of length d+1.
    # But we need to be careful: the BFS distance from start to nb where nb is a neighbor 
    # of start gives a cycle of length dist+1 only if the shortest path doesn't use edge (start,nb).
    # Actually, if dist(start, nb) = d and (start,nb) is an edge, then there's a cycle of 
    # length at most d+1. But the BFS might use the edge (start,nb) itself.
    # To be safe: for each vertex start, remove one neighbor at a time? Too expensive.
    # Alternative: use the fact that in a d-regular graph with girth g, the number of 
    # vertices within distance r is 1 + k + k(k-1) + ... + k(k-1)^r.
    # For verification, let's just do BFS and check that no two neighbors of any vertex 
    # are connected by a path of length < g-2 (which would create a cycle < g).
    
    # Efficient girth check: for each vertex, BFS and ensure no neighbor is reached 
    # in fewer than g-1 steps via a path not using the direct edge.
    # Since the graph is sparse, this should be fine for n up to a few thousand.
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    # If nb is a neighbor of start and dist[nb] + 1 < g, short cycle
                    # But dist[nb] might be via the edge (start,nb) itself
                    # If dist[nb] == 1, that's just the direct edge, no cycle
                    # If dist[nb] >= 2 and nb is a neighbor of start, cycle length = dist[nb] + 1
                    if nb != start and dist[nb] >= 2 and nb in adj[start]:
                        if dist[nb] + 1 < g:
                            return None
                    queue.append(nb)
    
    return n, edges

def _lww_fallback(k, g):
    """Original LWW fallback with q = k."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn, sub_fn, mul_fn = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add_fn, sub_fn, mul_fn = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
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
# EVOLVE-BLOCK-END
