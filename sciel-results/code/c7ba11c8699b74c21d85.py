"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency: for each independent vertex, which clique vertices it connects to
        cross_neighbors = []
        for v in range(k, n):
            neighbors = []
            for u in range(k):
                if rng.random() < p:
                    neighbors.append(u)
            cross_neighbors.append(neighbors)
        return cross_neighbors
    
    def clique_partition(n, k, p, seed):
        cross_neighbors = instance(n, k, p, seed)
        cliques = []
        
        # 1. The big clique on all k clique vertices
        cliques.append(list(range(k)))
        
        # 2. For each independent vertex, cover its cross edges
        for i, v in enumerate(range(k, n)):
            neighbors = cross_neighbors[i]
            if len(neighbors) >= 2:
                # One clique: {v} union its neighbors in C
                clique = list(neighbors)
                clique.append(v)
                cliques.append(clique)
            elif len(neighbors) == 1:
                # One edge: [neighbor, v]
                cliques.append([neighbors[0], v])
            # else: no cross edges, nothing to add
        
        return cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = clique_partition(n, k, p, seed)
        # Ensure all elements are Python ints and cliques are lists
        cleaned = []
        for c in cliques:
            cleaned.append([int(x) for x in c])
        result.append(cleaned)
    
    return result
# EVOLVE-BLOCK-END
