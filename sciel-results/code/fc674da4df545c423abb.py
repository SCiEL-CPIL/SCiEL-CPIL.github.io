"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        N = [set() for _ in range(k)]
        for u, v in edges:
            if u < k and v >= k:
                N[u].add(v)
            elif v < k and u >= k:
                N[v].add(u)
        
        if k > 0:
            common_neighbors = set(N[0])
            for u in range(1, k):
                common_neighbors &= N[u]
        else:
            common_neighbors = set()
        
        cliques = []
        
        clique_0 = list(range(k)) + list(common_neighbors)
        if len(clique_0) >= 2:
            cliques.append(clique_0)
        
        for u in range(k):
            if time.time() > deadline:
                break
            s_u = N[u] - common_neighbors
            if s_u:
                clique_u = [u] + list(s_u)
                cliques.append(clique_u)
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        result.append(solve_instance(*spec))
        
    return result
# EVOLVE-BLOCK-END
