def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # C is the clique of the first k vertices
        C = list(range(k))
        # I is the independent set of the rest
        I = list(range(k, n))
        
        cliques = []
        
        # 1. Cover edges within C using the clique C itself
        # C is a clique and has size k >= 2
        cliques.append(C)
        
        # 2. Cover cross edges
        # For each v in I, the edges incident to v are (u, v) for u in N(v)
        # These edges are disjoint for different v.
        # We can cover all edges incident to v with the clique N(v) U {v}.
        # This clique is valid because N(v) is a subset of C (a clique) and v is connected to all in N(v).
        for v in I:
            # Find neighbors of v in C
            neighbors = [u for u in C if (u, v) in edge_set or (v, u) in edge_set]
            if len(neighbors) > 0:
                # The clique is neighbors + v
                # Ensure it has at least 2 vertices (len(neighbors) >= 1 implies len(clique) >= 2)
                clique_v = neighbors + [v]
                cliques.append(clique_v)
                
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    return [clique_partition(*spec) for spec in specs]
