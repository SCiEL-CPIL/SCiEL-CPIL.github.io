"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs two disjoint K_{3,3} components.
    Each K_{3,3} has 6 vertices, degree 3, girth 4.
    Cycle lengths in K_{3,3} are 4 and 6.
    Total distinct cycle lengths: {4, 6} -> 2.
    """
    edges = []
    
    # First K_{3,3} on vertices 0,1,2 and 3,4,5
    part1 = [0, 1, 2]
    part2 = [3, 4, 5]
    for u in part1:
        for v in part2:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6,7,8 and 9,10,11
    part1 = [6, 7, 8]
    part2 = [9, 10, 11]
    for u in part1:
        for v in part2:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
