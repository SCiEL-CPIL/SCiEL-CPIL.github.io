def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_safe(n, k, p, seed, deadline):
        rng = Random(seed)
        # Build adjacency for core vertices (0..k-1) and cross edges
        # adj_core[u] is set of v in core connected to u
        adj_core = [set() for _ in range(k)]
        # adj_cross[i] is set of u in core connected to i (where i in k..n-1)
        adj_cross = [set() for _ in range(n)] 
        
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj_cross[v].add(u)
                    adj_core[u].add(v) # Not really needed for core-core, but good for symmetry if needed
                    # Actually, adj_core is for core-core edges which are always present.
        
        # Core edges are all pairs in 0..k-1
        # We don't need to generate them explicitly for existence, just for coverage tracking.
        
        # Set of core edges (u, v) with u < v, u, v < k
        # Initially all are available
        available_core_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        
        cliques = []
        
        # Process each vertex in I (k..n-1)
        for i in range(k, n):
            if time.time() > deadline:
                break
                
            N_i = adj_cross[i]
            if not N_i:
                continue
                
            # We want to find a set S subset of N_i such that all edges within S are in available_core_edges
            # To maximize coverage, we want S to be a large clique in the graph induced by N_i 
            # using only available_core_edges.
            
            # Greedy Maximal Clique in the graph G_avail
            # Vertices: N_i
            # Edges: (u, v) in G_avail if (min(u,v), max(u,v)) in available_core_edges
            
            S = set()
            # Sort N_i to ensure determinism and potentially better greedy choices (arbitrary here)
            for v in sorted(N_i):
                can_add = True
                for u in S:
                    edge = (min(u, v), max(u, v))
                    if edge not in available_core_edges:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            
            if S:
                # Form clique [i] + S
                cliques.append([i] + list(S))
                
                # Remove edges within S from available_core_edges
                S_list = list(S)
                for idx_u in range(len(S_list)):
                    for idx_v in range(idx_u + 1, len(S_list)):
                        u = S_list[idx_u]
                        v = S_list[idx_v]
                        available_core_edges.discard((min(u, v), max(u, v)))
                
                # The cross edges (i, c) for c in S are covered.
                # What about cross edges (i, c) for c in N_i \ S?
                # They are NOT covered by this clique.
                # We must cover them with 2-cliques [i, c]
                for c in N_i:
                    if c not in S:
                        cliques.append([i, c])
            else:
                # S is empty, so no core edges can be covered by a clique containing i.
                # Cover all cross edges with 2-cliques
                for c in N_i:
                    cliques.append([i, c])
        
        # If we broke out of the loop due to deadline, we must handle remaining I vertices and core edges
        # But the loop iterates through all I. If we break, we haven't processed the rest.
        # The problem states "Every edge must lie in exactly one listed clique".
        # If we don't process an i, its cross edges are uncovered.
        # So we must ensure we process all i.
        # The deadline check is just a safety. If we are close to deadline, we should still finish quickly.
        # The operations per i are small.
        
        # Now cover remaining core edges
        # available_core_edges contains edges not covered by any I-clique.
        # We need to partition these edges into cliques.
        # Greedy clique cover on the graph (C, available_core_edges)
        
        remaining_edges = set(available_core_edges)
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: use 2-cliques for all remaining
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Find a maximal clique in the graph defined by remaining_edges
            # Vertices involved in remaining_edges
            # We can just iterate through vertices 0..k-1
            
            S = set()
            for v in range(k):
                can_add = True
                for u in S:
                    edge = (min(u, v), max(u, v))
                    if edge not in remaining_edges:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            
            if len(S) < 2:
                # If S has 0 or 1 vertices, it covers no edges.
                # Pick an arbitrary edge from remaining_edges
                if remaining_edges:
                    u, v = next(iter(remaining_edges))
                    S = {u, v}
                else:
                    break
            
            if len(S) >= 2:
                cliques.append(list(S))
                
                # Remove edges within S from remaining_edges
                S_list = list(S)
                for idx_u in range(len(S_list)):
                    for idx_v in range(idx_u + 1, len(S_list)):
                        u = S_list[idx_u]
                        v = S_list[idx_v]
                        remaining_edges.discard((min(u, v), max(u, v)))
        
        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve_safe(*spec, deadline))

    return results
