def construct():
    # Fano plane on 7 vertices (0-6) is the unique 3-uniform hypergraph
    # with the minimum number of edges (7) that is exactly 3-chromatic.
    # Vertices 7 and 8 are isolated.
    
    # Build edges as a list of lists (each edge is a triple of distinct vertices)
    e1 = [0, 1, 2]
    e2 = [0, 3, 4]
    e3 = [0, 5, 6]
    e4 = [1, 3, 5]
    e5 = [1, 4, 6]
    e6 = [2, 3, 6]
    e7 = [2, 4, 5]
    
    edges = [e1, e2, e3, e4, e5, e6, e7]
    
    # Valid 3-coloring: every edge must be non-monochromatic
    # Vertex: 0  1  2  3  4  5  6  7  8
    # Color:  0  1  2  1  2  0  1  0  0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {"edges": edges, "colors": colors}
