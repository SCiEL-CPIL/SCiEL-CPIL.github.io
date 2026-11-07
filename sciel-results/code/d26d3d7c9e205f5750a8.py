def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices with chromatic number 3,
    minimizing the number of edges. The Fano plane (7 edges) on vertices 0-6
    is the known minimum. Vertices 7 and 8 are isolated.
    """
    # Fano plane edges on vertices 0..6
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Colors for the 9 vertices.
    # A valid 3-coloring of the Fano plane:
    # 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1
    # Vertices 7, 8 are isolated, can be any color.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 1]
    
    return {'edges': fano_edges, 'colors': colors}
