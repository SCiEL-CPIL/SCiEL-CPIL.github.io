"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Construct a 3-uniform hypergraph on 9 vertices that is exactly 3-chromatic
    with the minimum number of edges.
    
    We use the Fano plane, which is the unique 3-uniform hypergraph with 7 edges
    that is not 2-colorable. It is defined on the 7 non-zero vectors of GF(2)^3.
    Vertices 7 and 8 are isolated and assigned color 0.
    """
    # Define the 7 non-zero vectors of GF(2)^3 as integers 1 to 7
    # Mapping index i to vector (i+1) is not quite right, let's just use the values 1..7
    # Vector values: 1, 2, 3, 4, 5, 6, 7
    vectors = [1, 2, 3, 4, 5, 6, 7]
    
    edges = []
    # Generate edges: triples of distinct vectors that sum to 0 mod 2 (XOR to 0)
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                # Check if vectors[i] ^ vectors[j] ^ vectors[k] == 0
                if (vectors[i] ^ vectors[j] ^ vectors[k]) == 0:
                    edges.append([i, j, k])
    
    # A valid 3-coloring for the Fano plane vertices 0-6
    # Based on the assignment:
    # v0 (1): 0
    # v1 (2): 0
    # v2 (3): 1
    # v3 (4): 1
    # v4 (5): 2
    # v5 (6): 2
    # v6 (7): 0
    colors_7 = [0, 0, 1, 1, 2, 2, 0]
    
    # Pad with color 0 for vertices 7 and 8
    colors = colors_7 + [0, 0]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
