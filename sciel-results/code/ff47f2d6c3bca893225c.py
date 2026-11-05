def construct():
    # Use additive coloring modulo 15.
    # Vertices 0..15.
    # c(i, j) = (i + j) % 15
    # This uses 15 colors.
    # Validity:
    # A K4 has < 5 distinct colors iff some color appears >= 3 times.
    # For c(i,j) = (i+j) % m, 3 edges in a K4 sharing a color implies
    # either 3 vertices are congruent mod m, or two pairs of vertices are congruent 
    # in a specific way (a=d, b=c) which requires two distinct residue classes 
    # of size >= 2.
    # With m=15 and vertices 0..15, only residue 0 has size 2 (0, 15). 
    # All others have size 1.
    # Thus, we cannot have two distinct classes of size >= 2.
    # We also cannot have 3 vertices in the same class.
    # Therefore, no K4 has 3 edges of the same color.
    # So every K4 has at least 5 distinct colors.
    
    return [(i + j) % 15 for i in range(16) for j in range(i + 1, 16)]
