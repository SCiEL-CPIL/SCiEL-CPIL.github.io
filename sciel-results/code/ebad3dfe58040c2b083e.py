"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    # The optimal diameter-critical graph of diameter 2 on n vertices is the complete bipartite graph K_{n/2, n/2} for even n.
    # This graph is triangle-free, ensuring every edge is critical (removing it increases distance between endpoints to 3).
    mid = n // 2
    set_a = list(range(mid))
    set_b = list(range(mid, n))
    
    edges = []
    # Generate all edges between set_a and set_b
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
