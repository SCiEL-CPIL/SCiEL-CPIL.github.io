"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 16

def construct():
    """
    Constructs a simple undirected graph on vertices 0..15 with diameter exactly 2,
    where every edge is essential.
    
    The graph used is the complete bipartite graph K_{8,8}.
    Vertices 0..7 form partition A, and 8..15 form partition B.
    Edges exist between every pair (u, v) with u in A and v in B.
    
    Properties:
    1. Diameter is 2.
       - Any u in A, v in B are adjacent (distance 1).
       - Any u1, u2 in A have distance 2 (via any node in B).
       - Any v1, v2 in B have distance 2 (via any node in A).
       So diameter is 2.
    2. Every edge is essential.
       The condition that deleting an edge (u,v) increases the diameter above 2 or disconnects the graph
       is equivalent to requiring that the graph is triangle-free.
       In K_{8,8}, there are no triangles. For any edge (u,v), there is no common neighbor w.
       Thus, removing (u,v) leaves u and v with no path of length 2.
       Since the graph is bipartite and connected, removing an edge doesn't disconnect the graph
       (unless it's a bridge, but in K_{8,8} with parts size >= 2, no edge is a bridge).
       So the distance between u and v becomes >= 3.
       Thus, the diameter increases from 2 to at least 3.
    
    Returns:
        List of edges as [u, v] lists.
    """
    edges = []
    # Partition A: 0..7
    # Partition B: 8..15
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    # Ensure types are Python int
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
# EVOLVE-BLOCK-END
