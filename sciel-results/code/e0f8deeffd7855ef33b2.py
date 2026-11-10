"""Rewritten program for clique partition of split graphs."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        cliques = []
        
        # 1. Add the clique C itself to cover all edges within C.
        if k >= 2:
            cliques.append(list(range(k)))
            
        # 2. Add a 2-clique for every cross-edge (u, v) where u in C, v in I.
        for u, v in edges:
            if u < k and v >= k:
                cliques.append([u, v])
            elif v < k and u >= k:
                cliques.append([v, u])
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for u, v in edges:
                if u < k and v >= k:
                    cliques.append([u, v])
                elif v < k and u >= k:
                    cliques.append([v, u])
            results.append(cliques)
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
