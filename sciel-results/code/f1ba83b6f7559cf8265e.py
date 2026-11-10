"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a simple undirected graph on vertices 0..15 with diameter exactly 2,
    such that deleting ANY individual edge increases the diameter above 2 or disconnects the graph.
    Maximize the number of edges.
    
    Reasoning:
    1. For an edge (u, v) to be essential, its removal must prevent any path of length <= 2 between u and v.
       This implies u and v cannot have a common neighbor.
       If u and v had a common neighbor w, then (u, w) and (w, v) would form a path of length 2,
       so removing (u, v) would leave dist(u, v) = 2, failing the condition (diameter would not increase 
       above 2, nor would the graph necessarily disconnect).
       Therefore, the graph must be triangle-free.
    
    2. We want to maximize the number of edges in a triangle-free graph on n=16 vertices.
       By Mantel's Theorem, the maximum number of edges in a triangle-free graph is floor(n^2 / 4).
       For n=16, max edges = 16^2 / 4 = 64.
    
    3. The complete bipartite graph K_{8,8} achieves this bound.
       - Partition vertices into A = {0, ..., 7} and B = {8, ..., 15}.
       - Edges are all pairs (a, b) with a in A, b in B.
       - Total edges: 8 * 8 = 64.
       - Triangle-free: Yes, bipartite graphs contain no odd cycles.
       - Diameter: 
         - dist(a1, a2) for a1, a2 in A is 2 (via any b in B).
         - dist(b1, b2) for b1, b2 in B is 2 (via any a in A).
         - dist(a, b) for a in A, b in B is 1.
         - Max distance is 2.
       - Edge Essentiality:
         - Take any edge (a, b) with a in A, b in B.
         - Do a and b share a common neighbor? 
           - Neighbors of a are all vertices in B.
           - Neighbors of b are all vertices in A.
           - Intersection is empty.
         - Thus, removing (a, b) removes the only path of length 1.
         - No path of length 2 exists.
         - Shortest path becomes length 3 (e.g., a -> b' -> a' -> b).
         - Diameter increases to 3.
         - Condition satisfied.
    
    This construction is optimal.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 16
    mid = n // 2  # 8
    
    edges = []
    # Construct K_{8,8}
    for u in range(mid):
        for v in range(mid, n):
            # Ensure types are strictly Python int
            edges.append([int(u), int(v)])
            
    # Check deadline (trivially passed for this O(n^2) construction)
    if time.time() > deadline:
        return edges
        
    return edges
# EVOLVE-BLOCK-END
