"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct the complete bipartite graph K_{8,8} on vertices 0..15.
    This is the maximum edge count (64) for a triangle-free graph on 16 vertices (Mantel's Theorem).
    Triangle-free graphs with diameter 2 have the property that removing any edge
    increases the distance between its endpoints to >= 3 (as no alternative path of length 2 exists),
    thus increasing the diameter or disconnecting the graph.
    """
    n = 16
    half = n // 2
    # List comprehension is generally faster than append in a loop
    edges = [[u, v] for u in range(half) for v in range(half, n)]
    return edges

# EVOLVE-BLOCK-END
