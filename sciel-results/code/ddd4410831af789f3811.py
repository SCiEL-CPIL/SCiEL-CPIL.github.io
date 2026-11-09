"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """
    Constructs a complete bipartite graph K_{8,8}.
    This graph has:
    - 16 vertices
    - 64 edges
    - Diameter 2
    - Triangle-free, so removing any edge increases the distance between its endpoints to 3,
      thereby increasing the diameter to at least 3 (or disconnecting if n was smaller, but here it stays connected).
    """
    n = P['n']
    # Split vertices into two sets of size n/2
    mid = n // 2
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
