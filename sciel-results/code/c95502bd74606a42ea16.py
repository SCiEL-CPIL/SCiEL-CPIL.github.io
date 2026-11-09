def construct():
    import time
    start_time = time.time()
    DEADLINE = 140
    
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        C = list(range(k))
        I = list(range(k, n))
        
        cliques = []
        covered_internal = set()
        
        for v in I:
            neighbors_in_C = sorted([u for u in C if u in adj[v]])
            if neighbors_in_C:
                clique = [v] + neighbors_in_C
                cliques.append(clique)
                for i in range(len(neighbors_in_C)):
                    for j in range(i + 1, len(neighbors_in_C)):
                        covered_internal.add((neighbors_in_C[i], neighbors_in_C[j]))
        
        # Remaining internal C edges
        remaining_edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                if j in adj[i] and (i, j) not in covered_internal:
                    remaining_edges.add((i, j))
        
        # Greedy clique cover for remaining edges
        while remaining_edges:
            # Find a maximal clique in the remaining graph
            # Start with an edge
            u, v = next(iter(remaining_edges))
            clique = {u, v}
            # Try to expand
            candidates = adj[u] & adj[v]
            for w in sorted(candidates):
                if w not in clique and all(w in adj[x] for x in clique):
                    # Check all edges from w to clique are in remaining_edges
                    can_add = True
                    for x in clique:
                        edge = (min(w, x), max(w, x))
                        if edge not in remaining_edges:
                            can_add = False
                            break
                    if can_add:
                        clique.add(w)
            
            clique_list = sorted(clique)
            if len(clique_list) >= 2:
                cliques.append(clique_list)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        remaining_edges.discard((min(a, b), max(a, b)))
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() - start_time > DEADLINE:
            results.append([])
            continue
        cliques = solve_split_graph(*spec)
        results.append(cliques)
    
    return results
