"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    Each K_{3,3} is bipartite with partitions of size 3.
    This ensures:
    1. Minimum degree is 3.
    2. The graph is triangle-free (bipartite).
    3. The graph contains cycles (4-cycles and 6-cycles).
    4. The set of distinct cycle lengths is {4, 6}, so the raw value is 2.
    """
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partitions: {0, 1, 2} and {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Partitions: {6, 7, 8} and {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
