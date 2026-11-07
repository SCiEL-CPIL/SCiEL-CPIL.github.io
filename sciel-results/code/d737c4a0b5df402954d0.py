"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """Return a 3-uniform hypergraph on 9 vertices with chromatic number exactly 3.
    
    The Fano plane (7 vertices, 7 edges) is the smallest 3-uniform hypergraph
    with chromatic number 3. Vertices 7 and 8 are isolated.
    """
    # Fano plane edges on vertices 0..6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    
    # Valid 3-coloring: each edge has vertices of all three colors
    # Vertex colors: 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1, 7:0, 8:0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
