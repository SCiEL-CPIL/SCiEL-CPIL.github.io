import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    if k == 3 and g == 16:
        # Try the known optimal construction: 936 vertices
        # This is based on the Exoo et al. (2026) result
        # We'll use a voltage lift approach
        n, edges = _construct_936_girth16()
        if n > 0 and edges is not None:
            # Verify girth quickly
            if _verify_girth_nonbacktracking(n, edges, g):
                return n, edges
        # Fallback to LUW
        return _construct_luw_safe(k, g, deadline)
    else:
        return _construct_luw_safe(k, g, deadline)

def _construct_936_girth16():
    """
    Construct the known optimal 3-regular graph with girth 16 on 936 vertices.
    
    This uses a voltage lift of a base graph. The base graph is a 3-regular
    bipartite graph (the incidence graph of GQ(2,2) has 24 vertices and girth 8).
    We apply a voltage assignment using S_3 to get a 936-vertex graph.
    
    936 = 24 * 39, but 39 is not the order of S_3 (which is 6).
    So we need a different approach.
    
    Actually, let's use a direct algebraic construction.
    The graph can be constructed as a Cayley graph or using a specific
    permutation-based construction.
    
    For now, let's try a systematic search for a good voltage assignment
    on a smaller base graph.
    """
    # Try to construct using a known family
    # The 936-vertex graph might be constructible as a lift of a 12-vertex base
    # 936 / 12 = 78, still not a group order
    
    # Let's try a different approach: use the fact that we can construct
    # graphs using permutation groups
    # 
    # A 3-regular graph can be described by three permutations on n vertices
    # such that their product has no short cycles
    #
    # For n=936, we can try to find three permutations with the right properties
    #
    # Actually, let's use a simpler approach: try to construct the graph
    # using a specific known construction pattern
    
    # The Exoo et al. construction likely uses a specific algebraic family.
    # Without the exact paper, let's try a heuristic:
    # 1. Start with a random 3-regular graph on 936 vertices
    # 2. Use local search to increase girth
    # 
    # But this is unlikely to find girth 16 in 120s
    #
    # Alternative: use the LUW construction with a twist
    # LUW(3, 8) gives girth 16 but with ~6560 vertices
    # Can we find a quotient or subgraph with girth 16 and fewer vertices?
    
    # Let's try a different algebraic construction
    # Using the fact that 936 = 2^3 * 3 * 13
    
    # Actually, let me try a direct construction using a known method
    # for cage graphs: the "cage" construction using finite fields
    
    # For girth 16, we need a graph where the non-backtracking walks
    # of length < 8 don't return to the start
    
    # Let's try to use a specific construction:
    # The graph might be a covering of a smaller graph with a specific group
    
    # Given the complexity, let's return None and fall back to LUW
    # but with a better search strategy
    
    return 0, None

def _construct_luw_safe(k, g, deadline):
    """Construct LUW graph with safety deadline."""
    min_d = max(3, g // 2 - 1)
    for d in range(min_d, min_d + 8):
        if time.time() > deadline - 5:
            break
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
            if _verify_girth_nonbacktracking(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
    # Last resort: use larger d
    return _construct_luw_optimized(k, min_d + 8)

def _verify_girth_nonbacktracking(n, edges, g):
    """
    Verify that the graph has girth at least g using non-backtracking walks.
    A non-backtracking walk of length L starting at vertex v and ending at u
    with u = v implies a cycle of length <= L (actually, we need to be careful).
    
    More precisely: if there's a non-backtracking walk of length L from v back to v,
    then there's a cycle of length <= L. For girth >= g, we need no such walks
    for L < g.
    
    But actually, a non-backtracking closed walk of length L implies a cycle
    of length <= L. So if we check all non-backtracking walks of length < g
    from every vertex, and none return to the start, then girth >= g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do a BFS in the non-backtracking sense
    # State: (current_vertex, previous_vertex)
    # We track the minimum distance to return to start
    
    for start in range(n):
        # dist[prev][curr] = shortest non-backtracking path from (start, None) to (prev, curr)
        # But this is expensive. Let's use a simpler approach.
        
        # Actually, for girth checking, we can use the following:
        # A cycle of length < g exists iff there's a non-backtracking closed walk
        # of length < g.
        # 
        # We can check this by, for each vertex, doing a BFS where we track
        # (vertex, parent) states, and see if we can return to start in < g steps.
        
        # This is O(n * g * k) per start vertex, so O(n^2 * g * k) total
        # For n=6560, g=16, k=3: 6560^2 * 16 * 3 ≈ 2 billion, too slow
        
        # Better approach: use the standard girth algorithm
        # For each edge, remove it and BFS to find shortest path between its endpoints
        # This gives the shortest cycle through that edge
        # Total: O(n * m) = O(n^2 * k)
        
        # For n=6560, k=3: 6560^2 * 3 ≈ 130 million, feasible in Python with optimization
        
        pass
    
    # Use a faster girth check
    return _check_girth_fast(n, edges, g)

def _check_girth_fast(n, edges, g):
    """Fast girth check: for each edge, find shortest cycle through it."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each edge (u, v), temporarily remove it and find shortest path u to v
    # The shortest cycle through (u,v) has length = shortest_path(u,v) + 1
    
    # To speed up, we can do BFS from each vertex and track distances
    # But we need to avoid using the edge itself
    
    # Alternative: use the fact that for 3-regular graphs, we can use
    # a modified BFS that tracks the "first edge" used
    
    # Let's use a different approach: BFS from each vertex, and when we find
    # a back-edge, compute the cycle length
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_optimized(k, d):
    """Construct LUW (Lovasz-Ullman-Wold) graph."""
    if k != 3:
        return _construct_luw_generic(k, d)
    
    q = 3
    # Rule for LUW construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
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

def _construct_luw_generic(k, d):
    """Generic LUW construction for arbitrary k."""
    q = k
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
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

# EVOLVE-BLOCK-END
