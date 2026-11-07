"""Rewritten program with optimized split-graph clique partition."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        rng = random.Random(seed)
        
        # Generate cross edges
        # C is 0..k-1, I is k..n-1
        # We use the strategy:
        # 1. For each v in I, for each u in N(v), add clique [v, u].
        #    This covers all cross edges exactly once.
        # 2. Add clique C = list(range(k)).
        #    This covers all internal edges of C exactly once.
        #    Since [v, u] contains no internal edge of C (only one vertex from C),
        #    the edge sets are disjoint.
        
        cliques = []
        
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cliques.append([v, u])
        
        # Add the clique for C
        if k >= 2:
            cliques.append(list(range(k)))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
