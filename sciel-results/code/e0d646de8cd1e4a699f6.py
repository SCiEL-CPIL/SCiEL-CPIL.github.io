def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on vertices 0..15.
    
    This graph has 64 edges.
    1. Diameter is 2:
       - Vertices in the same partition are distance 2 apart (via any vertex in the other partition).
       - Vertices in different partitions are distance 1 apart (direct edge).
    2. Every edge is essential:
       - The graph is triangle-free.
       - For any edge (u, v), N(u) and N(v) are disjoint sets (they belong to opposite partitions).
       - Removing (u, v) leaves no path of length 2 between u and v.
       - The shortest path becomes length 3, increasing the diameter to 3.
    
    This is the maximum possible number of edges for such a graph, as proven by Mantel's Theorem 
    for triangle-free graphs.
    """
    n = 16
    # Partition vertices into two sets of 8
    A = list(range(0, 8))
    B = list(range(8, 16))
    
    edges = []
    for u in A:
        for v in B:
            # Ensure types are Python int
            edges.append([int(u), int(v)])
            
    return edges
