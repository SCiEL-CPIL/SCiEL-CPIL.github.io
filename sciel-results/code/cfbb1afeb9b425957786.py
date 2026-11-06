"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a C5-free graph with independence number < 9.
    Construction: 8 disjoint copies of K4.
    n = 32.
    alpha = 8.
    """
    n = 32
    # Generate edges for 8 disjoint K4s
    # Each K4 is on vertices [4*b, 4*b+1, 4*b+2, 4*b+3]
    edges = [
        [4 * b + i, 4 * b + j] 
        for b in range(8) 
        for i in range(4) 
        for j in range(i + 1, 4)
    ]
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
