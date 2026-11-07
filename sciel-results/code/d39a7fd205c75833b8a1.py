"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is a 3-uniform, 3-chromatic hypergraph with 7 edges on 7 vertices.
    # It is the theoretical minimum for 3-chromatic 3-uniform hypergraphs.
    # We embed it in 9 vertices (0..8), leaving vertices 7 and 8 isolated.
    
    # Fano plane edges on vertices 0..6
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Convert to lists as required
    edges = [list(e) for e in fano_edges]
    
    # A valid 3-coloring for the Fano plane vertices 0..6
    # c(0)=0, c(1)=0, c(2)=1, c(3)=1, c(4)=2, c(5)=2, c(6)=0
    # Verification:
    # {0,1,2}: 0,0,1 (non-mono)
    # {0,3,4}: 0,1,2 (non-mono)
    # {0,5,6}: 0,2,0 (non-mono)
    # {1,3,5}: 0,1,2 (non-mono)
    # {1,4,6}: 0,2,0 (non-mono)
    # {2,3,6}: 1,1,0 (non-mono)
    # {2,4,5}: 1,2,2 (non-mono)
    
    colors_0_6 = [0, 0, 1, 1, 2, 2, 0]
    
    # Vertices 7 and 8 are isolated, assign color 0
    colors_7_8 = [0, 0]
    
    colors = colors_0_6 + colors_7_8
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
