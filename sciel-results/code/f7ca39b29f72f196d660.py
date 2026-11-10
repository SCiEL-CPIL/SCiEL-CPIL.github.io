"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    deadline = time.time() + 140.0  # safety margin before the 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build the exact same split graph as the seed generator.
        # Clique on first k vertices, independent set on the rest.
        # Cross pairs joined with probability p.
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def build_partition(cross_edges):
        """
        A valid clique partition where every edge lies in exactly one clique:
        - One big clique on the first k vertices (covers all K_k edges).
        - Each cross edge (u, v) as its own 2-clique.
        Total cliques = 1 + |cross_edges|.
        This is strictly better than the seed's k*(k-1)/2 + |cross_edges|.
        """
        cliques = [list(range(k))]
        for (u, v) in cross_edges:
            cliques.append([int(u), int(v)])
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > deadline:
            # Fallback: just use the big clique + all cross edges as 2-cliques
            # (same logic, but we should not reach here given the small sizes)
            ce = instance(n, k, p, seed)
            result.append(build_partition(ce))
        else:
            ce = instance(n, k, p, seed)
            result.append(build_partition(ce))

    # Ensure all types are Python int and lists
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
# EVOLVE-BLOCK-END
