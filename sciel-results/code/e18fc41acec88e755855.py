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
        # C is {0, ..., k-1}.
        if k >= 2:
            cliques.append(list(range(k)))
            
        # 2. Add a 2-clique for every cross-edge (u, v) where u in C, v in I.
        # This covers each cross-edge exactly once.
        # It does not cover any C-C edge (which are already covered by C).
        # It does not cover any I-I edge (there are none).
        for u, v in edges:
            if u < k and v >= k:
                cliques.append([u, v])
            elif v < k and u >= k:
                cliques.append([v, u])
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: return empty or minimal valid structure if timeout
            # Though this logic is O(E), it should be very fast.
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Just return the edges as cliques if we run out of time? 
            # No, that might be invalid if C-C edges are included as 2-cliques 
            # and C clique is missing. 
            # Given the speed, this branch is unlikely.
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
