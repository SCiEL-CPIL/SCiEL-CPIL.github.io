"""Combined program: deterministic construction of two disjoint K3,3 graphs to minimize distinct cycle lengths."""
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Returns edges for two disjoint K_{3,3} graphs on vertices 0..11.
    K_{3,3} is 3-regular, bipartite (girth 4), and has only cycle lengths 4 and 6.
    Distinct cycle lengths: {4, 6} -> Count = 2.
    This is likely the minimum possible for min-degree 3, girth >= 4.
    """
    n = 12
    edges = []
    
    # First K_{3,3}: vertices 0,1,2 (Part A) and 3,4,5 (Part B)
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Second K_{3,3}: vertices 6,7,8 (Part A) and 9,10,11 (Part B)
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    # Ensure types are Python int
    result = []
    for e in edges:
        u, v = e
        if isinstance(u, int) and isinstance(v, int) and not isinstance(u, bool) and not isinstance(v, bool):
            result.append([int(u), int(v)])
            
    return result

# EVOLVE-BLOCK-END
