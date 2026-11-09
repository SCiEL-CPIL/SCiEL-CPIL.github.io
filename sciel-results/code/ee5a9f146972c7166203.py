"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 9, 'r': 3, 'colors': 3}

def construct():
    # The Fano plane on vertices 0..6 is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges.
    # Vertices 7 and 8 are isolated, so they do not affect the chromatic number.
    # We provide a valid 3-coloring for the Fano plane and assign color 0 to isolated vertices.
    
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
    # c0=0, c1=1, c2=2, c3=0, c4=1, c5=1, c6=0
    # Isolated vertices 7, 8 get color 0.
    colors = [0, 1, 2, 0, 1, 1, 0, 0, 0]
    
    return {'edges': [list(e) for e in edges], 'colors': colors}
# EVOLVE-BLOCK-END
