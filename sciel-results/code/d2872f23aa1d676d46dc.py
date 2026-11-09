"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K4: no 5-cycle and independence number 8.
    # This achieves n=32, which is the known lower bound for R(C5, K9).
    n_cliques = 8
    clique_size = 4
    n = n_cliques * clique_size
    edges = []
    for b in range(n_cliques):
        base = b * clique_size
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
