def construct():
    """
    Returns the Fano plane (7 edges) on 9 vertices (0-6 active, 7-8 isolated).
    This is the minimal 3-chromatic 3-uniform hypergraph.
    """
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 1]
    
    # Verification (low cost)
    for e in fano_edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Monochromatic edge found")
            
    return {'edges': fano_edges, 'colors': colors}
