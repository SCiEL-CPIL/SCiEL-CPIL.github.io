"""Clique partition of split graphs using structural properties."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Strategy:
        # 1. One clique containing all vertices in the clique part (0..k-1).
        #    This covers all edges within the clique part exactly once.
        # 2. For every edge between the independent part (k..n-1) and the clique part,
        #    create a 2-clique.
        #    This covers all cross edges exactly once.
        #    Since these 2-cliques contain only one vertex from the clique part,
        #    they do not cover any edges within the clique part, avoiding conflicts.
        
        cliques = []
        
        # 1. The main clique for the clique part
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. 2-cliques for cross edges
        for u, v in edges:
            if (u < k and v >= k) or (u >= k and v < k):
                # Ensure consistent ordering if needed, but list just needs vertices
                # The task says "list of at least two distinct int vertices"
                cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        # Ensure types are Python int and list
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
