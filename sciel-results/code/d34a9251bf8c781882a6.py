"""C5-free graph with independence number < 9; target n=32 (8x K4)."""
# EVOLVE-BLOCK-START
def construct():
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
