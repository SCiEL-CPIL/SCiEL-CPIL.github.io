"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 12-vertex graph with minimum degree >= 3, girth >= 4 (triangle-free),
    that minimizes the number of distinct simple cycle lengths.
    
    Strategy: Two disjoint K_{3,3} components.
    K_{3,3} has exactly two distinct cycle lengths: 4 and 6.
    This gives raw value = 2.
    """
    edges = []
    
    # First component: K3,3 on vertices 0-5
    # Part A: {0, 1, 2}
    # Part B: {3, 4, 5}
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
    
    # Second component: K3,3 on vertices 6-11
    # Part A: {6, 7, 8}
    # Part B: {9, 10, 11}
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
