"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Define the two components of the graph: two disjoint K3,3
    # Component 1: Vertices 0,1,2 connected to 3,4,5
    # Component 2: Vertices 6,7,8 connected to 9,10,11
    
    # Generate edges for first K3,3
    A1 = np.array([0, 1, 2])
    B1 = np.array([3, 4, 5])
    edges1 = []
    for u in A1:
        for v in B1:
            edges1.append([int(u), int(v)])
            
    # Generate edges for second K3,3
    A2 = np.array([6, 7, 8])
    B2 = np.array([9, 10, 11])
    edges2 = []
    for u in A2:
        for v in B2:
            edges2.append([int(u), int(v)])
            
    # Combine
    return edges1 + edges2
# EVOLVE-BLOCK-END
