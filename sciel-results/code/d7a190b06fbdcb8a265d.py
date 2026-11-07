"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # The optimal graph is the complete bipartite graph K_{8,8}.
    # Partition vertices into two sets of size 8.
    # Set A: 0..7
    # Set B: 8..15
    # Edges connect every vertex in A to every vertex in B.
    # This graph is triangle-free, has diameter 2, and every edge is critical.
    # Total edges = 8 * 8 = 64.
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
