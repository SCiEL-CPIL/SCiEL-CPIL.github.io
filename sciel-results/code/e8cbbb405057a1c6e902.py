def construct():
    # Fano plane: 7 edges on vertices 0-6, provably 3-chromatic (not 2-colorable)
    # Vertices 7, 8 are isolated (no edges incident to them)
    # This is the minimum possible edge count for a 3-chromatic 3-uniform hypergraph
    
    # Standard Fano plane edges
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Convert to list of lists with Python int types
    edges = [list(e) for e in fano_edges]
    
    # Valid 3-coloring: no edge is monochromatic
    # Vertices 0-6 colored with 3 colors, vertices 7,8 get color 0
    color_labels = (0, 1, 1, 2, 2, 1, 2, 0, 0)
    colors = [int(c) for c in color_labels]
    
    # Verify every edge is non-monochromatic
    for e in edges:
        c = [colors[v] for v in e]
        if len(set(c)) == 1:
            raise ValueError(f"Edge {e} is monochromatic")
    
    return {
        'edges': edges,
        'colors': colors
    }
