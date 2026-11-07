def construct():
    # The Fano plane is the smallest 3-uniform hypergraph that is 3-chromatic.
    # It consists of 7 vertices and 7 edges.
    # We embed it in the 9-vertex space (0..8) using vertices 0..6.
    # Vertices 7 and 8 are isolated and do not affect the chromatic number.
    # This configuration achieves the minimum possible edge count of 7.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6.
    # We must ensure no edge is monochromatic.
    # Coloring:
    # v0: 0
    # v1: 1
    # v2: 2
    # v3: 1
    # v4: 2
    # v5: 0
    # v6: 1
    #
    # Verification of non-monochromaticity:
    # [0, 1, 2] -> [0, 1, 2] (OK)
    # [0, 3, 4] -> [0, 1, 2] (OK)
    # [0, 5, 6] -> [0, 0, 1] (OK)
    # [1, 3, 5] -> [1, 1, 0] (OK)
    # [1, 4, 6] -> [1, 2, 1] (OK)
    # [2, 3, 6] -> [2, 1, 1] (OK)
    # [2, 4, 5] -> [2, 2, 0] (OK)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are strictly Python int and list to satisfy type checks
    # and avoid NumPy type issues.
    edges_out = [list(e) for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {
        'edges': edges_out,
        'colors': colors_out
    }
