# EVOLVE-BLOCK-START
def construct():
    # 8 disjoint copies of K4
    # n = 32
    # Independence number is 8 (1 from each K4)
    # No 5-cycles (K4 has no 5-cycles, and components are disjoint)
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
