def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        edges = instance(n, k, p, seed)
        
        neighbors_in_C = {v: [] for v in range(k, n)}
        
        for u, v in edges:
            if v >= k:
                neighbors_in_C[v].append(u)
                
        cliques = []
        
        # Add the main clique C
        # k is at least 10, so size >= 2
        cliques.append(list(range(k)))
        
        # Add cliques for each vertex in I with neighbors in C
        for v in range(k, n):
            nbrs = neighbors_in_C[v]
            if nbrs:
                cliques.append([v] + nbrs)
                
        results.append(cliques)
        
    return results
