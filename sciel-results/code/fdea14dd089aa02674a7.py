"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s limit

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        cliques = []
        
        # Clique on all C vertices covers all C-C edges
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Each cross edge as a 2-clique
        for (u, v) in sorted(edges):
            if u < k and v >= k:
                cliques.append([u, v])
            elif v < k and u >= k:
                cliques.append([v, u])
            # C-C edges are already covered by the big clique
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            break
        results.append(solve_instance(*spec))
    
    while len(results) < 3:
        results.append([])
    
    return results
# EVOLVE-BLOCK-END
