import itertools

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Strategy: Use 8 disjoint copies of K4.
    n = 32.
    alpha = 8 (one vertex from each K4).
    C5-free: K4 has no C5, and components are disjoint.
    """
    n = 32
    edges = []
    # Create 8 disjoint cliques of size 4
    for block_idx in range(8):
        base = block_idx * 4
        # Generate all pairs in the clique
        vertices_in_block = [base + i for i in range(4)]
        for u, v in itertools.combinations(vertices_in_block, 2):
            edges.append([u, v])
    return {'n': n, 'edges': edges}
