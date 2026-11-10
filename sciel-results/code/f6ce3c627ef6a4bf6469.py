"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_instance(n, k, p, seed):
        rng = Random(seed)
        # Generate cross edges: for each independent vertex v in [k, n), 
        # determine which clique vertices it connects to
        cross_neighbors = []
        for v in range(k, n):
            neighbors = []
            for u in range(k):
                if rng.random() < p:
                    neighbors.append(u)
            cross_neighbors.append((v, neighbors))
        
        cliques = []
        
        # 1 clique for all edges within the clique {0, ..., k-1}
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each independent vertex with at least one neighbor, 
        # form a clique N(v) U {v}
        for v, neighbors in cross_neighbors:
            if len(neighbors) >= 1:
                clique = sorted(neighbors + [v])
                cliques.append(clique)
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        # Convert to ensure pure Python ints and lists
        cliques = [[int(x) for x in clique] for clique in cliques]
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
