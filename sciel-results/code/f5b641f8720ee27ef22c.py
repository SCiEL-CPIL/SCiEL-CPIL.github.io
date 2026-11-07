"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns the edges of the complete bipartite graph K_{8,8}.
    This graph has 64 edges, diameter 2, and is triangle-free.
    For any triangle-free graph with diameter 2, every edge is critical
    for maintaining diameter <= 2, because removing an edge (u,v) leaves
    u and v at distance >= 3 (or disconnected), as there is no triangle
    to provide a length-2 path.
    By Mantel's Theorem, K_{8,8} is the unique triangle-free graph on 16
    vertices with the maximum number of edges (64).
    """
    n = 16
    h = n // 2
    out = []
    for a in range(h):
        for b in range(h, n):
            out.append([a, b])
    return out

# EVOLVE-BLOCK-END
