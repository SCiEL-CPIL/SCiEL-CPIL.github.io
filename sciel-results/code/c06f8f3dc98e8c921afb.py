"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices (0..8) that is 
    3-chromatic (not 2-colorable) with the minimum number of edges.
    
    The Fano plane (7 edges) is the smallest 3-uniform hypergraph that is not 2-colorable.
    We embed the Fano plane on vertices 0..6 and leave vertices 7 and 8 isolated.
    This preserves non-2-colorability: any 2-coloring of the 9-vertex hypergraph 
    restricts to a 2-coloring of the Fano plane, which is impossible.
    """
    # The Fano plane edges (7 edges on 7 vertices)
    # This is a 3-uniform hypergraph where every edge has exactly 3 vertices.
    # It is known to be the smallest 3-uniform hypergraph that is not 2-colorable.
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
    # Isolated vertices 7, 8 can be any color; we choose 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verification: ensure no monochromatic edge in the provided coloring
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            raise ValueError("Invalid coloring: monochromatic edge found")
    
    # Ensure types are exactly Python int (not bool, not numpy int)
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
