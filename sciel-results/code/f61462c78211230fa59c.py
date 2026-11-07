"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique minimal 3-chromatic 3-uniform hypergraph.
    # It has 7 vertices and 7 edges.
    # We embed it on vertices 0..6. Vertices 7 and 8 are isolated.
    # This achieves the theoretical minimum edge count of 7.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verified 3-coloring for the Fano plane:
    # v0: 0, v1: 1, v2: 2
    # v3: 1, v4: 2, v5: 0, v6: 2
    # v7: 0, v8: 0 (isolated)
    
    colors = [0, 1, 2, 1, 2, 0, 2, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
