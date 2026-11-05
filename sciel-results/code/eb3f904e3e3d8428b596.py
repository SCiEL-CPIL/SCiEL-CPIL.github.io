"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # 8 disjoint K4s plus one extra vertex connected to one vertex in each K4.
    # n = 33, C5-free, alpha = 8 < 9.
    n = 33
    edges = []
    # 8 K4s: vertices 0-31
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([4 * b + i, 4 * b + j])
    # Vertex 32 connected to vertex 4*b in each K4
    for b in range(8):
        edges.append([32, 4 * b])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
