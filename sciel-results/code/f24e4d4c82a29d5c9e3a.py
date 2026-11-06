import time

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    The edge (i, j) with i < j is assigned color i.
    This is a valid Gallai coloring (no rainbow 4-cycles) because 
    in any 4-cycle, the vertex with the smallest index connects to 
    two other vertices, and both edges incident to it will have the 
    same color (the smallest index). Thus, at least two edges in the 
    cycle share a color, preventing 4 distinct colors.
    The number of distinct colors is N-1 = 15.
    """
    colors = []
    for i in range(N):
        for j in range(i + 1, N):
            colors.append(i)
    return colors
# EVOLVE-BLOCK-END
