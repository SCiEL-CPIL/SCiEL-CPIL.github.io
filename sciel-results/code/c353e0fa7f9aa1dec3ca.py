"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices with the minimum number of edges
    such that it is 3-chromatic (not 2-colorable, but 3-colorable).
    
    The Fano plane (7 vertices, 7 edges) is the smallest 3-uniform hypergraph
    that is not 2-colorable, so 7 edges is the theoretical minimum.
    We embed it in the 9-vertex space {0..8} using vertices 0..6.
    """
    
    # Fano plane edges on vertices 0-6 (projective plane of order 2)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring: no edge is monochromatic
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
