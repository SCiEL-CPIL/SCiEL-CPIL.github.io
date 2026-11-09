import time

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Strategy: Use 8 disjoint copies of K4.
    n = 8 * 4 = 32.
    alpha = 8 * 1 = 8 < 9.
    C5-free: K4 has no 5-cycle (only cycles of length 3 and 4),
    and disjoint components cannot form cycles spanning multiple components.
    
    This is the known best construction for R(C5, K9) > 32.
    """
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
