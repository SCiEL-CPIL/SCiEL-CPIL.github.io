"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Separate K-internal edges and cross edges
        k_edges = []
        cross_edges = []
        for (u, v) in edges:
            if u < k and v < k:
                k_edges.append((u, v))
            else:
                cross_edges.append((u, v))
        
        cliques = []
        # One clique for all of K (covers all K-internal edges)
        if k >= 2:
            cliques.append(list(range(k)))
        # Each cross edge gets its own clique {u, v}
        for (u, v) in cross_edges:
            cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 5:
            # Fallback: just do the simple thing quickly
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for (u, v) in edges:
                if not (u < k and v < k):
                    cliques.append([u, v])
            results.append(cliques)
        else:
            results.append(clique_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
