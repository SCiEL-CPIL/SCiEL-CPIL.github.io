"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency: for each independent-set vertex, which clique vertices it connects to
        # Cross edges: (u, v) where u in [0,k), v in [k,n)
        adj = {}
        for v in range(k, n):
            neighbors = [u for u in range(k) if rng.random() < p]
            adj[v] = neighbors
        return adj

    def make_cliques(n, k, p, seed):
        adj = instance(n, k, p, seed)
        cliques = []
        # The full clique on the first k vertices (covers all intra-clique edges)
        if k >= 2:
            cliques.append(list(range(k)))
        # For each independent-set vertex, {v} ∪ N(v) is a clique covering its cross-edges
        for v in range(k, n):
            nbrs = adj[v]
            if len(nbrs) >= 1:
                cliques.append([v] + nbrs)
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = make_cliques(n, k, p, seed)
        # Ensure all are lists of Python ints
        cliques = [[int(x) for x in c] for c in cliques]
        results.append(cliques)
    return results
# EVOLVE-BLOCK-END
