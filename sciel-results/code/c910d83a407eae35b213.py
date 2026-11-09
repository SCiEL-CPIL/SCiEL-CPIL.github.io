"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano Plane is a 3-uniform hypergraph on 7 vertices with 7 edges.
    # It is 3-chromatic (not 2-colorable, but 3-colorable).
    # We embed it on vertices 0..6 and leave 7,8 isolated to satisfy the 9-vertex constraint.
    # This achieves the theoretical minimum edge count of 7.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano Plane embedded above.
    # c[0]=0, c[1]=1, c[2]=2, c[3]=1, c[4]=0, c[5]=2, c[6]=1
    # Vertices 7 and 8 are isolated, so their color doesn't affect edge constraints.
    colors = [0, 1, 2, 1, 0, 2, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
