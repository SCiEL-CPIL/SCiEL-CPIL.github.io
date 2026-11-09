"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 disjoint copies of K4.
    - Each K4 has 4 vertices and 6 edges.
    - K4 does not contain a 5-cycle (C5).
    - The independence number of K4 is 1.
    - For 8 disjoint copies, total n = 32, total alpha = 8.
    - Since alpha = 8, there is no independent set of size 9.
    - Since components are K4, there are no 5-cycles.
    - n=32 is the current best known lower bound for R(C5, K9).
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
