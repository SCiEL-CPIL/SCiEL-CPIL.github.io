"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    
    - Vertices 0, 1, 2 are connected to 3, 4, 5 (First K_{3,3}).
    - Vertices 6, 7, 8 are connected to 9, 10, 11 (Second K_{3,3}).
    
    Properties:
    - Simple undirected graph.
    - Every vertex has degree 3.
    - Bipartite, so triangle-free (girth 4).
    - Contains cycles of length 4 and 6.
    - Distinct cycle lengths: {4, 6}. Count = 2.
    """
    edges = []
    
    # First K_{3,3} component: partitions {0, 1, 2} and {3, 4, 5}
    part1_left = [0, 1, 2]
    part1_right = [3, 4, 5]
    for u in part1_left:
        for v in part1_right:
            edges.append([u, v])
            
    # Second K_{3,3} component: partitions {6, 7, 8} and {9, 10, 11}
    part2_left = [6, 7, 8]
    part2_right = [9, 10, 11]
    for u in part2_left:
        for v in part2_right:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
