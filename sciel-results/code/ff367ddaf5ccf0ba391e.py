def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
        
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Build adjacency for I
        # neighbors[i] stores list of neighbors in C
        neighbors = [[] for _ in range(n)]
        
        for u, v in edges:
            if u < k and v < k:
                # Internal edge, ignore for now, handled by C clique
                pass
            else:
                # u < k, v >= k
                neighbors[v].append(u)
                
        cliques = []
        
        # Add the main clique C
        if k >= 2:
            cliques.append(list(range(k)))
            
        # Process independent set vertices
        for i in range(k, n):
            nb = neighbors[i]
            if len(nb) >= 1:
                # If len(nb) >= 1, nb + [i] is a valid clique
                # If len(nb) == 1, it's a size-2 clique
                # If len(nb) >= 2, it's a star clique covering all incident edges
                # This is always optimal as argued
                cliques.append(nb + [i])
                
        results.append(cliques)
        
    return results
