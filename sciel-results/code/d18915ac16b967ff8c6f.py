# EVOLVE-BLOCK-START
def construct():
    # The optimal graph for maximizing edges in a diameter-2 graph 
    # where every edge is critical (removal increases diameter) 
    # is the complete bipartite graph K_{8,8}.
    # 
    # Reasoning:
    # 1. If an edge (u, v) is in a triangle (u, v, w), then removing (u, v) 
    #    leaves a path u-w-v of length 2. If the rest of the graph is 
    #    well-connected (diameter 2), the diameter of the remaining graph 
    #    remains 2, meaning the edge was not critical.
    # 2. Therefore, to ensure every edge is critical, the graph should be 
    #    triangle-free.
    # 3. By Mantel's Theorem, the maximum number of edges in a triangle-free 
    #    graph on n vertices is floor(n^2 / 4).
    # 4. For n=16, max edges = 16^2 / 4 = 64.
    # 5. K_{8,8} achieves this bound and has diameter 2.
    
    n = 16
    partition_a = set(range(0, 8))
    partition_b = set(range(8, 16))
    
    edges = []
    # Iterate through all pairs in A and B to form complete bipartite edges
    for a in partition_a:
        for b in partition_b:
            edges.append([a, b])
            
    # Ensure return type is list of lists of integers
    return edges
# EVOLVE-BLOCK-END
