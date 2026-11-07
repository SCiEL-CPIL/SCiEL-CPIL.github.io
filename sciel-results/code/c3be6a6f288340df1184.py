"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges.
    # It is known to be the smallest non-2-colorable 3-uniform hypergraph.
    # We embed it in 9 vertices by leaving vertices 7 and 8 unused.
    # The edges are the lines of the Fano plane.
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring of the Fano plane vertices.
    # Colors for vertices 0..6: 0, 1, 2, 0, 1, 2, 0
    # Vertices 7 and 8 are unused, so their color doesn't matter. We set them to 0.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Convert edges to lists of Python ints
    edges_list = [list(e) for e in edges]
    
    return {'edges': edges_list, 'colors': colors}

# EVOLVE-BLOCK-END
