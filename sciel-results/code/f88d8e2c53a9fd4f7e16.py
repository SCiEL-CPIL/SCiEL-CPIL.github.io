"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The Fano plane is the unique (up to isomorphism) 3-uniform hypergraph 
    # on 7 vertices with 7 edges that is 3-chromatic. 
    # It is proven that no 3-uniform hypergraph with fewer than 7 edges 
    # is 3-chromatic (all such hypergraphs are 2-colorable).
    # Thus, 7 is the minimum possible number of edges.
    
    # Vertices 0..6 form the Fano plane. Vertices 7 and 8 are isolated 
    # (not part of any edge), which is allowed as the task specifies 
    # "triples on 0..8" but doesn't require all vertices to be incident 
    # to an edge, only that the edges are distinct triples on the set.
    # The coloring must be valid for the edges.
    
    # Define the Fano plane edges as tuples of vertex indices
    fano_edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    )
    
    # Verify non-monochromaticity with the chosen coloring
    # Vertices: 0, 1, 2, 3, 4, 5, 6
    # Colors:   0, 1, 2, 1, 2, 0, 1
    vertex_colors = [0, 1, 2, 1, 2, 0, 1]
    
    for edge in fano_edges:
        colors_in_edge = [vertex_colors[v] for v in edge]
        if len(set(colors_in_edge)) == 1:
            raise ValueError(f"Edge {edge} is monochromatic!")
    
    # Assign colors to all 9 vertices (7, 8 are isolated, can be any color)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Return the result with proper types
    return {
        'edges': list(fano_edges),
        'colors': list(colors)
    }
# EVOLVE-BLOCK-END
