"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
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
    
    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Find neighbors in the clique for each independent vertex
        # And determine which independent vertices are universal (connected to all k clique vertices)
        neighbor_sets = {}  # v -> set of clique neighbors
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (min(u,v), max(u,v)) in edges:
                    nbrs.add(u)
            if nbrs:
                neighbor_sets[v] = nbrs
        
        # Universal vertices: connected to all k clique vertices
        universal = set()
        for v, nbrs in neighbor_sets.items():
            if len(nbrs) == k:
                universal.add(v)
        
        # Build cliques
        cliques = []
        
        # Main clique: all clique vertices + universal independent vertices
        main_clique = list(range(k)) + sorted(universal)
        if len(main_clique) >= 2:
            cliques.append(main_clique)
        
        # For each non-universal independent vertex with neighbors, add its clique
        for v in sorted(neighbor_sets.keys()):
            if v in universal:
                continue
            nbrs = neighbor_sets[v]
            clique = [v] + sorted(nbrs)
            if len(clique) >= 2:
                cliques.append(clique)
        
        return cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        result.append(partition_cliques(*spec))
    
    return result
# EVOLVE-BLOCK-END
