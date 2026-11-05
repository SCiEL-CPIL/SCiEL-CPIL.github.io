"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K4: no 5-cycle and independence number 8.
    # This maximizes n given alpha < 9, as K4 is the largest C5-free clique.
    n_blocks = 8
    vertices_per_block = 4
    n = n_blocks * vertices_per_block
    edges = []
    for b in range(n_blocks):
        base = b * vertices_per_block
        for i in range(vertices_per_block):
            for j in range(i + 1, vertices_per_block):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
