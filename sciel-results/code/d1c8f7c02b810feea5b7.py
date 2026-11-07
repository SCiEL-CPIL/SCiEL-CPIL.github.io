def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        # cross edges: for v in independent set [k, n), for u in [0,k), edge if rng.random() < p
        # Must match seed generator exactly: edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((u, v))
        return cross

    result = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cross = instance(n, k, p, seed)
        
        # Identify neighbors for each u in C
        # neighbors[u] will be a list of v in I such that (u, v) is an edge
        neighbors = [[] for _ in range(k)]
        for (u, v) in cross:
            neighbors[u].append(v)
            
        cliques = []
        # 1. The clique on C
        cliques.append(list(range(k)))
        
        # 2. For each u in C, the clique {u} U N_I(u)
        for u in range(k):
            if neighbors[u]:
                clique = [int(u)] + [int(v) for v in neighbors[u]]
                cliques.append(clique)
            # If neighbors[u] is empty, no clique is needed for u's cross edges (there are none)
            
        result.append(cliques)
    return result
