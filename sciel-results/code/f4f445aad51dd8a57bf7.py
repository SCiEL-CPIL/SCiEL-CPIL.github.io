"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K4: no 5-cycle and independence number 8.
    # n = 8 * 4 = 32.
    # This certifies R(C5, K9) > 32.
    return {'n': 32, 'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]}
# EVOLVE-BLOCK-END
