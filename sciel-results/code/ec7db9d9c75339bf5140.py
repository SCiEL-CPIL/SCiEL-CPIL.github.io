"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3,
    triangle-free (girth >= 4), and minimizes distinct simple cycle lengths.
    
    Strategy: Use two disjoint copies of K_{3,3}.
    - K_{3,3} is 3-regular.
    - K_{3,3} is bipartite, so no odd cycles.
    - Smallest cycle in K_{3,3} is length 4.
    - K_{3,3} contains 6-cycles.
    - Does K_{3,3} contain 8-cycles? No, max cycle length is 6 (Hamiltonian).
    - So distinct cycle lengths are {4, 6}. Count = 2.
    
    This is better than the seed prism graph which has {4, 6, 8, 12}.
    """
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partitions: {0, 1, 2} and {3, 4, 5}
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Partitions: {6, 7, 8} and {9, 10, 11}
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    # Ensure types are correct (list of lists of ints)
    # The problem states: "Return edges as [u, v] lists"
    # "integers must be Python int"
    
    return edges

# EVOLVE-BLOCK-END
