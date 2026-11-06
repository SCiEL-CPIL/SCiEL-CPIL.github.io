"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    The edges are ordered lexicographically by (i, j) with 0 <= i < j < 16.
    Uses the coloring scheme color(i, j) = i.
    This ensures that for any 4-cycle, the vertex with the smallest index 
    has two incident edges with the same color (its own index), 
    preventing the cycle from having 4 distinct colors.
    This uses N-1 = 15 distinct colors.
    """
    colors = []
    for i in range(N):
        for j in range(i + 1, N):
            colors.append(i)
    return colors
# EVOLVE-BLOCK-END
