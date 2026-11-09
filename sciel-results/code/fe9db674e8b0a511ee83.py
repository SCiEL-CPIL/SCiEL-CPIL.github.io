import time

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Uses 8 disjoint copies of K4.
    n = 8 * 4 = 32.
    alpha = 8 * 1 = 8 < 9.
    K4 contains no C5, and disjoint union preserves C5-freeness.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well within 120s limit

    n_cliques = 8
    clique_size = 4
    n = n_cliques * clique_size  # 32
    edges = []

    for b in range(n_cliques):
        if time.time() > deadline:
            break
        base = b * clique_size
        for i in range(clique_size):
            if time.time() > deadline:
                break
            for j in range(i + 1, clique_size):
                edges.append([base + i, base + j])

    return {'n': n, 'edges': edges}
