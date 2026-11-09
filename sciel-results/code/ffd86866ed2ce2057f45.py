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
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane:
    # Vertices: 0, 1, 2, 3, 4, 5, 6
    # Colors:   0, 1, 2, 1, 2, 0, 1
    # Let's verify non-monochromaticity:
    # (0,1,2) -> 0,1,2 OK
    # (0,3,4) -> 0,1,2 OK
    # (0,5,6) -> 0,0,1 OK
    # (1,3,5) -> 1,1,0 OK
    # (1,4,6) -> 1,2,1 OK
    # (2,3,6) -> 2,1,1 OK
    # (2,4,5) -> 2,2,0 OK
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are strictly Python int and list
    edges_list = [tuple(int(v) for v in e) for e in edges]
    colors_list = [int(c) for c in colors]
    
    return {'edges': edges_list, 'colors': colors_list}
# EVOLVE-BLOCK-END
