    def construct():
        from random import Random
        import time
        
        def instance(n, k, p, seed):
            rng = Random(seed)
            edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
            edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
            return sorted(edges)
            
        results = []
        for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            
            # Build sets
            edge_set = set(edges)
            # Neighborhoods for I in C
            neigh_C_in_I = {v: set() for v in range(k, n)}
            for u, v in edges:
                if u < k and v >= k:
                    neigh_C_in_I[v].add(u)
                elif v < k and u >= k:
                    neigh_C_in_I[u].add(v)
            
            cliques = []
            covered_pairs = set()
            
            # Step 1: Cross cliques
            for v in range(k, n):
                N = neigh_C_in_I[v]
                if N:
                    clique = [v] + list(N)
                    cliques.append(clique)
                    # Mark pairs in N as covered
                    N_list = list(N)
                    for i in range(len(N_list)):
                        for j in range(i+1, len(N_list)):
                            u1, u2 = N_list[i], N_list[j]
                            if u1 > u2: u1, u2 = u2, u1
                            covered_pairs.add((u1, u2))
            
            # Step 2: Remaining edges in C
            remaining_edges = []
            for u in range(k):
                for w in range(u+1, k):
                    if (u, w) in edge_set:
                        if (u, w) not in covered_pairs:
                            remaining_edges.append((u, w))
            
            # Step 3: Cover remaining edges
            # remaining_edges is a list of tuples
            # Convert to set for fast lookup
            rem_set = set(remaining_edges)
            
            while rem_set:
                # Pick an edge
                u, v = rem_set.pop()
                # Try to extend to a clique
                # Common neighbors of u and v in rem_set
                common = set()
                # Find neighbors of u in rem_set
                # This is O(E) per step, but E is small
                neighbors_u = {w for (a, b) in rem_set if a == u or b == u for w in (b if a == u else a)}
                neighbors_v = {w for (a, b) in rem_set if a == v or b == v for w in (b if a == v else a)}
                common = neighbors_u & neighbors_v
                
                if common:
                    w = next(iter(common))
                    clique = [u, v, w]
                    # Remove edges (u,w), (v,w)
                    rem_set.discard((min(u, w), max(u, w)))
                    rem_set.discard((min(v, w), max(v, w)))
                else:
                    clique = [u, v]
                
                cliques.append(clique)
            
            results.append(cliques)
        return results
    