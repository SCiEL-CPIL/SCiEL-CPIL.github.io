"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns the edge list of K_{8,8} (complete bipartite graph with 8+8 vertices).
    
    Properties:
    - 16 vertices (0..15)
    - 64 edges
    - Diameter exactly 2
    - Every edge is critical: removing any edge increases diameter to 3
      (the two endpoints of the removed edge no longer have a common neighbor,
       since the graph is triangle-free)
    
    This achieves the theoretical maximum for triangle-free diameter-2 graphs
    on 16 vertices (Mantel's theorem: max triangle-free edges = n^2/4 = 64).
    """
    n = 16
    half = n // 2
    edges = []
    for i in range(half):
        for j in range(half, n):
            edges.append([i, j])
    return edges
# EVOLVE-BLOCK-END
