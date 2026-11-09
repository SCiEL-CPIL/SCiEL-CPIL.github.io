# EVOLVE-BLOCK-START
def construct():
    # K_{8,8} is the unique triangle-free graph on 16 vertices with maximum edges (64).
    # By Mantel's theorem, no triangle-free graph on 16 vertices can have more than 64 edges.
    # Triangle-free + diameter 2 implies every edge is critical: removing edge (u,v)
    # leaves no common neighbor for u and v, so dist(u,v) >= 3, making diameter >= 3.
    # K_{8,8} has diameter exactly 2.
    # 64 is the theoretical maximum; this construction achieves it.
    edges = [[a, b] for a in range(8) for b in range(8, 16)]
    return edges
# EVOLVE-BLOCK-END
