"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        
        # Separate edges into internal (C-C) and cross (C-I)
        # C is 0..k-1, I is k..n-1
        internal_edges = []
        cross_edges = []
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.append((u, v))
            else:
                # One is < k, one is >= k
                cross_edges.append((u, v))
        
        cliques = []
        
        # 1. Cover all internal edges with the single clique C
        # C is the set of vertices 0 to k-1
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. Cover all cross edges with cliques of size 2
        # Each cross edge (u, v) is a clique {u, v}
        for u, v in cross_edges:
            cliques.append([u, v])
            
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
