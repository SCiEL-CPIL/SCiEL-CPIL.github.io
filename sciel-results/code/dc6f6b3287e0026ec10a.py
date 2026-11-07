"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import sys

def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3, triangle-free,
    minimizing the number of distinct simple cycle lengths.
    
    Strategy: Use two disjoint copies of K_3,3.
    K_3,3 is bipartite (triangle-free), 3-regular, and has cycle lengths {4, 6}.
    Thus, the distinct cycle lengths are {4, 6}, count = 2.
    This is likely optimal as min degree 3 graphs typically force multiple cycle lengths.
    """
    start_time = time.time()
    deadline = start_time + 60.0
    
    # Partitions for first K_3,3: {0, 1, 2} and {3, 4, 5}
    # Partitions for second K_3,3: {6, 7, 8} and {9, 10, 11}
    
    edges = []
    
    # First K_3,3
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_3,3
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Ensure types are Python int and list
    final_edges = []
    for e in edges:
        if time.time() > deadline:
            break
        # Explicitly cast to Python int to avoid numpy/bool issues
        u = int(e[0])
        v = int(e[1])
        final_edges.append([u, v])
        
    return final_edges

# EVOLVE-BLOCK-END
