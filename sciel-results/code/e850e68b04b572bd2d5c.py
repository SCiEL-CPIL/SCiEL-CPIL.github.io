"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    edges = []
    
    # Partition vertices into A (size 6) and B (size 10)
    A = list(range(6))
    B = list(range(6, 16))
    
    # Add all edges between A and B (Complete Bipartite K_{6,10})
    for a in A:
        for b in B:
            edges.append([a, b])
    
    # Add 5 disjoint edges inside B to reach 65 total edges
    # This creates triangles (a, b1, b2) with degrees 10, 7, 7 -> Sum 24
    internal_edges = [
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15]
    ]
    
    for u, v in internal_edges:
        edges.append([u, v])
        
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
