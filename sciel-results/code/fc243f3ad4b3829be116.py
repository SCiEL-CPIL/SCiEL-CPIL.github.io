def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        # Replicate RNG logic to get neighbors efficiently
        # Note: instance() does not use rng for K-K edges, so we start rng at the same state
        rng = Random(seed)
        
        neighbors = {v: set() for v in range(k, n)}
        
        # The instance function iterates:
        # for v in range(k, n):
        #   for u in range(k):
        #     if rng.random() < p:
        #       add edge
        # We do the same to populate neighbors.
        
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].add(u)
                    
        cliques = []
        covered_K_edges = set()
        
        # Step 1: Cliques for I vertices
        for v in range(k, n):
            N_v = neighbors[v]
            if N_v:
                # Create clique N_v U {v}
                # Ensure distinct vertices (guaranteed by set + v not in N_v)
                clique_v = list(N_v)
                clique_v.append(v)
                cliques.append(clique_v)
                
                # Mark K-edges covered by this clique
                N_v_list = list(N_v)
                for i in range(len(N_v_list)):
                    u = N_v_list[i]
                    for j in range(i + 1, len(N_v_list)):
                        w = N_v_list[j]
                        if u > w:
                            u, w = w, u
                        covered_K_edges.add((u, w))
                        
        # Step 2: Cover remaining K-edges
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_K_edges:
                    cliques.append([u, w])
                    
        return cliques

    return [solve_instance(*spec) for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))]
