"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices that is not 2-colorable.
    # It has 7 edges.
    # We extend it to 9 vertices by adding two isolated vertices (7 and 8).
    # This preserves the non-2-colorability (if a 2-coloring existed for 9 vertices,
    # its restriction to the first 7 would be a 2-coloring for the Fano plane, which is impossible).
    # The minimum number of edges for a 3-chromatic 3-uniform hypergraph is 7.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices (0-6)
    # c(0)=0, c(1)=1, c(2)=2, c(3)=1, c(4)=2, c(5)=0, c(6)=1
    # Isolated vertices 7, 8 can be any color, we choose 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
