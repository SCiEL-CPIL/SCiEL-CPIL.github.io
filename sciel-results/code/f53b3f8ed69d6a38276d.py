def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        remaining_edges = set(edges)
        
        # Generate candidate maximal cliques
        # In a split graph, maximal cliques are:
        # 1. The clique C itself
        # 2. {v} U N(v) for each v in I (independent set)
        
        C = list(range(k))
        candidates = [C]
        
        for v in range(k, n):
            neighbors_in_C = []
            for u in range(k):
                e = (u, v) if u < v else (v, u)
                if e in edges:
                    neighbors_in_C.append(u)
            if neighbors_in_C:
                clique = [v] + neighbors_in_C
                candidates.append(clique)
        
        cliques = []
        
        # Greedy selection: pick the clique covering the most remaining edges
        while remaining_edges:
            best_clique = None
            best_count = 0
            
            for clique in candidates:
                count = 0
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        e = (u, w) if u < w else (w, u)
                        if e in remaining_edges:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None or best_count == 0:
                # This should not happen if the candidates cover all edges
                # But if it does, we must handle it to avoid infinite loop
                # Pick any remaining edge and form a clique of size 2?
                # But that might not be a maximal clique.
                # The task requires "Every listed set must be a clique".
                # It does NOT require maximal.
                # So we can fallback to picking an edge.
                e = next(iter(remaining_edges))
                u, w = e
                best_clique = [u, w]
                best_count = 1
            
            cliques.append(best_clique)
            
            # Remove edges covered by this clique
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, w = best_clique[i], best_clique[j]
                    e = (u, w) if u < w else (w, u)
                    remaining_edges.discard(e)
        
        results.append(cliques)
    
    return results
