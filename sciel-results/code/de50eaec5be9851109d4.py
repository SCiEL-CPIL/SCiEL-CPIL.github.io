"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

P = {'n': 16}

def construct():
    """
    Returns K_{8,8}, which is the optimal solution.
    
    Key insight: An edge (u,v) is critical (removing it increases diameter > 2 or disconnects)
    if and only if u and v have NO common neighbor. This is because:
    - If they have a common neighbor w, then path u-w-v of length 2 exists, so dist(u,v) <= 2
      even after removing edge (u,v). The edge is NOT critical.
    - If they have no common neighbor, then after removing (u,v), the shortest path between
      u and v must be length >= 3 (or non-existent), so diameter > 2. The edge IS critical.
    
    Therefore, we need a triangle-free graph with diameter exactly 2.
    By Turan's theorem, the maximum number of edges in a triangle-free graph on n vertices
    is floor(n^2/4), achieved by the complete bipartite graph K_{n/2, n/2}.
    
    For n=16: floor(16^2/4) = 64 edges, achieved by K_{8,8}.
    K_{8,8} has diameter 2 (same part: dist 2, different parts: dist 1).
    Every edge in K_{8,8} is between parts, and no two edges share a common neighbor
    (bipartite property), so every edge is critical.
    
    This is the theoretical maximum, so 64 edges is optimal.
    """
    n = P['n']
    half = n // 2
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
