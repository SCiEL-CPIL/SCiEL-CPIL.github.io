import time
import sys

def construct():
    """
    Constructs a graph on vertices 0..11 consisting of two disjoint K_3,3 graphs.
    This ensures minimum degree 3, triangle-free (bipartite), and contains cycles.
    The distinct cycle lengths in K_3,3 are 4 and 6.
    Thus, the number of distinct cycle lengths is 2.
    """
    edges = set()
    
    # First K_3,3 on vertices 0,1,2 and 3,4,5
    part1 = [0, 1, 2]
    part2 = [3, 4, 5]
    for u in part1:
        for v in part2:
            edges.add(tuple(sorted((u, v))))
            
    # Second K_3,3 on vertices 6,7,8 and 9,10,11
    part3 = [6, 7, 8]
    part4 = [9, 10, 11]
    for u in part3:
        for v in part4:
            edges.add(tuple(sorted((u, v))))
            
    # Convert to list of lists
    return [list(e) for e in sorted(edges)]

# EVOLVE-BLOCK-END
