"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 16}
START_TIME = time.time()
TIME_LIMIT = 85

def construct():
    """
    Construct the complete bipartite graph K_{8,8} on vertices 0..15.
    
    This graph has:
    - 64 edges.
    - Diameter exactly 2.
    - Every edge is critical: removing any edge (u, v) increases the distance
      between u and v to 3 (via any other vertex in the opposite partition),
      thus increasing the diameter to 3.
    
    This is the maximum number of edges possible for a graph on 16 vertices
    where every edge is critical for diameter 2.
    """
    n = P['n']
    half = n // 2
    edges = []
    
    # Construct K_{half, n-half}
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
            
    # Safety deadline check (construction is O(n^2) and very fast, 
    # but we include it per directive)
    if time.time() - START_TIME > TIME_LIMIT:
        return edges

    return edges
# EVOLVE-BLOCK-END
