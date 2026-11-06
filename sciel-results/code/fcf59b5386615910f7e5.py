"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 40
    set_a = list(range(20))
    set_b = list(range(20, 40))
    
    # Strategy: Star construction in Set A to balance book sizes.
    # Add K intra-edges incident to vertex 0 in A.
    # Remove R bipartite edges incident to vertex 0 in A.
    # Constraint: K - R = 1.
    # Optimal balance for max book size is around R=9, K=10 or R=10, K=11.
    # We choose R=9, K=10.
    
    R = 9
    K = 10
    
    # Intra-edges: (0, i) for i in 1..K
    # Removed bipartite edges: (0, 20+j) for j in 0..R-1
    
    removed_edges = set()
    for j in range(R):
        removed_edges.add((0, 20 + j))
        
    edges = []
    
    # Add bipartite edges, skipping removed ones
    for u in set_a:
        for v in set_b:
            if (u, v) not in removed_edges:
                edges.append([u, v])
                
    # Add intra-edges
    for i in range(1, K + 1):
        edges.append([0, i])
        
    return edges
# EVOLVE-BLOCK-END
