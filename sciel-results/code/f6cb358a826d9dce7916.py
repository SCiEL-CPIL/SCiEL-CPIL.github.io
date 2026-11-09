import time

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Uses 8 disjoint copies of K4.
    n = 8 * 4 = 32.
    alpha = 8 * 1 = 8.
    Since 8 < 9, the condition is met.
    K4 contains no C5, so the union contains no C5.
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
