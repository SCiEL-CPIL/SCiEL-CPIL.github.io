"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_3,3 components.
    
    Properties:
    - Vertex set: {0, ..., 11}
    - Component 1: K_3,3 with partitions {0, 1, 2} and {3, 4, 5}
    - Component 2: K_3,3 with partitions {6, 7, 8} and {9, 10, 11}
    - Minimum degree: 3 (every vertex in K_3,3 has degree 3)
    - Girth: 4 (K_3,3 is bipartite, so no odd cycles; smallest cycle is 4)
    - Cycle lengths: The only simple cycles in K_3,3 have lengths 4 and 6.
      - 4-cycles: e.g., 0-3-1-4-0
      - 6-cycles: e.g., 0-3-1-4-2-5-0
      - No cycles of other lengths exist in K_3,3 (max cycle length is 6).
    - Total distinct cycle lengths: {4, 6}, count = 2.
    
    Theoretical Note:
    It is a known result that any graph with minimum degree at least 3 must contain
    cycles of at least two distinct lengths. Thus, a count of 2 is the global minimum.
    """
    edges = []
    
    # Component 1: K_3,3 on vertices 0, 1, 2 (left) and 3, 4, 5 (right)
    # We explicitly list edges to ensure they are Python ints and lists.
    # Left partition: 0, 1, 2
    # Right partition: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_3,3 on vertices 6, 7, 8 (left) and 9, 10, 11 (right)
    # Left partition: 6, 7, 8
    # Right partition: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
