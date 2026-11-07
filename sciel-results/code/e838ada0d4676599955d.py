"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        
        # N[u] is list of neighbors in K for u in I
        # Must match random call order: for v in range(k, n) for u in range(k)
        N = {}
        for u in range(k, n):
            neighbors = []
            for v in range(k):
                if rng.random() < p:
                    neighbors.append(v)
            N[u] = neighbors
            
        # Check for universal vertex in I (connected to all K)
        universal_u = None
        for u in range(k, n):
            if len(N[u]) == k:
                universal_u = u
                break
        
        cliques = []
        
        if universal_u is not None:
            # Use K union {universal_u}
            big_clique = list(range(k)) + [universal_u]
            cliques.append(big_clique)
            
            # For other u in I
            for u in range(k, n):
                if u == universal_u:
                    continue
                if N[u]:
                    clique = [u] + N[u]
                    cliques.append(clique)
        else:
            # Use K
            big_clique = list(range(k))
            cliques.append(big_clique)
            
            # For all u in I
            for u in range(k, n):
                if N[u]:
                    clique = [u] + N[u]
                    cliques.append(clique)
                    
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            break
        results.append(instance(*spec))
        
    return results
# EVOLVE-BLOCK-END
