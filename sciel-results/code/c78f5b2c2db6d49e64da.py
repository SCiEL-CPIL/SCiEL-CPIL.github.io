# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = 16
    # Complete bipartite graph K_{8,8}
    # Partition: A = {0, ..., 7}, B = {8, ..., 15}
    # Every vertex in A is connected to every vertex in B.
    # Edges: 8 * 8 = 64
    #
    # This achieves the Mantel bound: max edges in a triangle-free graph on 16 vertices
    # is floor(16^2/4) = 64. Since every edge being essential requires the graph to be
    # triangle-free (any triangle would make the third edge non-essential), 64 is optimal.
    #
    # Diameter is exactly 2:
    #   - Same partition vertices u,v in A: path u-w-v for any w in B. Dist = 2.
    #   - Opposite partition vertices u in A, v in B: direct edge. Dist = 1.
    #
    # Every edge is essential:
    #   - Graph is bipartite, hence triangle-free. For any edge (u,v) with u in A, v in B,
    #     u and v share no common neighbor. Removing (u,v): shortest path u to v is
    #     u -> b -> a' -> v (length 3) where b in B\{v}, a' in A\{u}.
    #     Diameter becomes 3 > 2. Edge is essential.
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])

    if time.time() > deadline:
        # Should never happen for this O(64) construction, but guard just in case
        pass

    return edges
# EVOLVE-BLOCK-END
