def construct():
    # The Fano plane is the smallest 3-chromatic 3-uniform hypergraph.
    # It has 7 edges. This is the theoretical minimum for this problem.
    # We use vertices 0..6 for the Fano plane and 7,8 are isolated.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Static lookup of a valid 3-coloring for the Fano plane vertices 0..6.
    # Coloring: [0, 1, 2, 0, 1, 2, 0]
    # Vertices 7 and 8 are isolated, so any valid color (0..2) works. We use 0.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    return {'edges': [list(e) for e in edges], 'colors': colors}
