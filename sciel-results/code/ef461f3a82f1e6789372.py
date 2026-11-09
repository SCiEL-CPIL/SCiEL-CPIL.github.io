"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph with 8 disjoint copies of K4.
    n = 32.
    Each K4 has alpha = 1.
    Total alpha = 8.
    K4 contains no 5-cycle.
    Disjoint union contains no 5-cycle.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
