# EVOLVE-BLOCK-START

def construct():
    n = 16
    # The optimal structure is the complete bipartite graph K_{8,8}.
    # This graph is triangle-free, which is a necessary condition for 
    # every edge to be essential (removing any edge increases diameter 
    # or disconnects). Mantel's theorem limits triangle-free graphs on 
    # 16 vertices to at most 64 edges, and K_{8,8} achieves this bound.
    
    # Partition vertices into two sets of size 8
    set_a = set(range(8))
    set_b = set(range(8, 16))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
