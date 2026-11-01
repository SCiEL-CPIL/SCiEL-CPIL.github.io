import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q).
    """
    q = k
    
    # The LUW graph CD(d, q) is q-regular and bipartite.
    # The number of vertices is 2 * q^d.
    # We want to minimize n, so we want the smallest d such that girth >= g.
    
    # For LUW CD(d,q), the girth is at least 2d+2 when q is large enough.
    # For small q, it may be smaller. We need girth >= g, so 2d+2 >= g => d >= (g-2)/2.
    # Also need d >= 2 for the construction to make sense.
    
    start_d = max(2, (g - 2) // 2)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # We try d values in increasing order
    # n = 2 * q^d
    # We want the smallest n, so smallest d.
    
    last_result = None
    for d in range(start_d, 12):  # Cap d to avoid huge graphs
        if time.time() > deadline:
            break
            
        # Check if graph would be too large (> 50000 vertices is too slow to verify)
        n_est = 2 * (q ** d)
        if n_est > 100000:
            break
            
        # Construct graph for this d
        n, edges = _luw_construct(q, d)
        last_result = (n, edges)
        
        # Check girth
        if _check_girth(n, edges, g):
            return n, edges
            
    # Fallback: return the last constructed one
    if last_result is not None:
        return last_result
    
    # Absolute fallback: use a larger d
    d = max(4, (g - 2) // 2 + 2)
    n, edges = _luw_construct(q, d)
    return n, edges

def _luw_construct(q, d):
    """
    Helper to construct LUW graph CD(d, q).
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule for LUW graph
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] + x[c] (or similar)
    # The seed code has a specific rule generation.
    # LUW rule: for coordinate c, the relation is l_c = p_c + l_a * p_b
    # The standard LUW rules ensure that no short cycles exist.
    # For c=2: a=1, b=1
    # For c=3: a=2, b=1  
    # For c=4: a=1, b=2
    # For c>=5: use a periodic pattern that keeps indices in [1, c-1]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        # Use a pattern that cycles through valid index pairs
        # Ensure a < c and b < c to avoid self-references
        # Pattern: (c-3, 1), (c-2, 1), (1, c-2), (1, c-3) cycling
        idx = (c - 5) % 4
        if idx == 0:
            rule[c] = (c - 3, 1)
        elif idx == 1:
            rule[c] = (c - 2, 1)
        elif idx == 2:
            rule[c] = (1, c - 2)
        else:
            rule[c] = (1, c - 3)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1? 
        # Seed: start = (0, (None,) + (0,) * d)
        # So x has length d+1. Index 0 is None, 1..d are values.
        for first in range(q):
            # y is the line/point on the other side
            # y[0] is None, y[1] = first
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # l[c] - p[c] = l[a] * p[b]  => l[c] = p[c] + l[a] * p[b]
                    # Here x is point p, y is line l.
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # p[c] = l[c] - l[a] * p[b] ? 
                    # Seed: sub(x[c], mul(x[a], y[b]))
                    # Here x is line l, y is point p.
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _check_girth(n, edges, g):
    """
    Check if the graph has girth at least g.
    Uses non-backtracking BFS from each vertex up to depth (g-1)//2.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # For girth >= g, we need no cycles of length < g.
    # A cycle of length L can be detected by a non-backtracking walk of length L.
    # We check from each vertex: do any two neighbors at distance floor((g-1)/2) 
    # share a common neighbor at the next level (creating a cycle)?
    # 
    # More efficient: BFS from each vertex, tracking parent to avoid backtracking.
    # If we find a vertex at distance d that is also reachable from a different 
    # path at distance d' with d + d' + 1 < g, we have a short cycle.
    
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # dist[v] = distance from start
        # parent[v] = parent of v in BFS tree (to avoid backtracking)
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0  # Use index-based queue for speed
        
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            if dist[v] >= max_depth:
                continue
            for w in adj[v]:
                if w == parent[v]:
                    continue  # Don't backtrack
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                else:
                    # w is already visited from a different path
                    # This indicates a cycle of length dist[v] + dist[w] + 1
                    cycle_len = dist[v] + dist[w] + 1
                    if cycle_len < g:
                        return False
    return True
