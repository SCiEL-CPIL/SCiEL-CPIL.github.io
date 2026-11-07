"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique 3-uniform hypergraph with 7 edges
    # that is 3-chromatic. Any 3-uniform hypergraph with <= 6 edges is 2-colorable.
    # Thus, 7 is the minimum number of edges.
    # We use vertices 0..6 for the Fano plane and leave 7, 8 isolated.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Valid 3-coloring for the Fano plane:
    # 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1
    # Check edges:
    # (0,1,2) -> 0,1,2 (ok)
    # (0,3,4) -> 0,1,2 (ok)
    # (0,5,6) -> 0,0,1 (ok)
    # (1,3,5) -> 1,1,0 (ok)
    # (1,4,6) -> 1,2,1 (ok)
    # (2,3,6) -> 2,1,1 (ok)
    # (2,4,5) -> 2,2,0 (ok)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Convert to required types: list of lists and list of ints
    edges_out = [list(e) for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
