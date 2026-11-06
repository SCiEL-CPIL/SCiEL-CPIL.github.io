"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K4: no 5-cycle and independence number 8.
    # This certifies R(C5, K9) > 32.
    num_blocks = 8
    block_size = 4
    n = num_blocks * block_size
    edges = []
    for b in range(num_blocks):
        for i in range(block_size):
            for j in range(i + 1, block_size):
                u = b * block_size + i
                v = b * block_size + j
                edges.append([u, v])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
