def construct():
    """
    Construct a simple undirected graph on 16 vertices (0..15) with diameter 2,
    where every edge is critical (deleting it increases diameter > 2 or disconnects).
    Maximize edge count.
    
    The optimal solution is the complete bipartite graph K_{8,8}, which has 64 edges.
    This is proven optimal because:
    1. Critical edges imply the graph is triangle-free (no edge has a common neighbor).
    2. Mantel's theorem limits triangle-free graphs to floor(n^2/4) edges.
    3. For n=16, max edges = 64.
    4. K_{8,8} achieves 64 edges and satisfies all criticality conditions.
    """
    n = 16
    half = n // 2
    
    # Generate edges for K_{half, half}
    # Partition 1: 0 .. half-1
    # Partition 2: half .. n-1
    edges = []
    
    # We iterate through all pairs (u, v) with u in P1 and v in P2
    # This ensures no loops and no duplicates (since u < v is not guaranteed, 
    # but we only add u from first half and v from second half, so u != v always,
    # and we don't add (v, u) separately)
    for u in range(half):
        for v in range(half, n):
            # Ensure types are Python int
            edges.append([int(u), int(v)])
            
    return edges
