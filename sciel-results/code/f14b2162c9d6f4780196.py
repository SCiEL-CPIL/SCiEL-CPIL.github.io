import time

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3, triangle-free,
    minimizing the number of distinct simple cycle lengths.
    
    The best known minimum number of distinct cycle lengths for such a graph is 2.
    We construct two disjoint copies of K_3,3.
    K_3,3 is 3-regular, bipartite (triangle-free), and has cycles of length 4 and 6.
    Thus, the set of cycle lengths is {4, 6}, so the count is 2.
    """
    edges = []
    
    # First K_3,3: Vertices 0,1,2 and 3,4,5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([int(u), int(v)])
            
    # Second K_3,3: Vertices 6,7,8 and 9,10,11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([int(u), int(v)])
            
    return edges

# EVOLVE-BLOCK-END
