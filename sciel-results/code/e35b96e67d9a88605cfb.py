def construct():
    from random import Random
    import time
    import sys

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _max_clique_in_subgraph(vertices, adj_matrix):
        """
        Find a maximum clique in the subgraph induced by 'vertices' 
        using the adjacency matrix 'adj_matrix' (where adj_matrix[u][v] is True if edge exists).
        Returns a list of vertices in the clique.
        """
        if not vertices:
            return []
        
        # Simple branch and bound for small graphs
        best_clique = []
        
        def expand(candidate, available):
            nonlocal best_clique
            if not available:
                if len(candidate) > len(best_clique):
                    best_clique = candidate[:]
                return
            
            # Greedy heuristic: pick vertex with highest degree in available
            # Or just iterate
            for i, v in enumerate(available):
                new_candidate = candidate + [v]
                # New available are neighbors of v in the remaining available
                new_available = [u for u in available[i+1:] if adj_matrix[v][u]]
                expand(new_candidate, new_available)
                
        expand([], list(vertices))
        return best_clique

    def _solve_safe(n, k, p, seed, deadline):
        rng = Random(seed)
        
        # Build adjacency for core vertices (0..k-1) and cross edges
        # adj_core[u] is set of v in core connected to u (always True for u!=v in core)
        # adj_cross[i] is set of u in core connected to i (where i in k..n-1)
        
        adj_cross = [set() for _ in range(n)]
        
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj_cross[v].add(u)
        
        # Core is a clique, so all edges (u, v) for u < v < k exist.
        # available_core_edges is a set of tuples (u, v) with u < v
        available_core_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        
        cliques = []
        
        # Pre-compute adjacency matrix for core for fast lookup
        # core_adj[u][v] is True if (u, v) is in available_core_edges
        # Initially all True. We will update this as we remove edges.
        core_adj = [[True] * k for _ in range(k)]
        for u in range(k):
            core_adj[u][u] = False
            
        def get_core_neighbors_in_set(S):
            """
            Given a set S of core vertices, find the largest subset T of S
            such that all edges within T are in available_core_edges.
            """
            if len(S) <= 1:
                return list(S)
            
            # Use the max clique finder on the subgraph induced by S
            # But we need to respect available_core_edges.
            # Let's build a temporary adjacency list for the subgraph
            S_list = list(S)
            m = len(S_list)
            # temp_adj[i][j] is True if edge between S_list[i] and S_list[j] is available
            temp_adj = [[False] * m for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    if i != j:
                        u = S_list[i]
                        v = S_list[j]
                        a, b = (u, v) if u < v else (v, u)
                        temp_adj[i][j] = (a, b) in available_core_edges
            
            # Find max clique in temp_adj
            # Simple recursive search
            best = []
            
            def dfs(candidate, available_indices):
                nonlocal best
                if not available_indices:
                    if len(candidate) > len(best):
                        best = candidate[:]
                    return
                
                # Heuristic: sort available by degree in the current candidate context?
                # Just iterate
                for idx, i in enumerate(available_indices):
                    new_candidate = candidate + [S_list[i]]
                    new_available = [j for j in available_indices[idx+1:] if temp_adj[i][j]]
                    dfs(new_candidate, new_available)
                    
            dfs([], list(range(m)))
            
            # Convert indices back to vertices
            return [S_list[i] for i in best]

        # Process each vertex in I (k..n-1)
        # Sort by degree descending to handle high-degree vertices first?
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda v: len(adj_cross[v]), reverse=True)
        
        for i in i_vertices:
            if time.time() > deadline:
                break
                
            N_i = adj_cross[i]
            if not N_i:
                continue
                
            uncovered_cross = set(N_i)
            
            while uncovered_cross:
                if time.time() > deadline:
                    # Fallback: cover remaining with 2-cliques
                    for c in uncovered_cross:
                        cliques.append([i, c])
                    break
                
                # Find largest clique S in uncovered_cross using available core edges
                S = get_core_neighbors_in_set(uncovered_cross)
                
                if len(S) >= 2:
                    cliques.append([i] + S)
                    # Remove edges within S from available_core_edges
                    S_list = list(S)
                    for idx_u in range(len(S_list)):
                        for idx_v in range(idx_u + 1, len(S_list)):
                            u = S_list[idx_u]
                            v = S_list[idx_v]
                            a, b = (u, v) if u < v else (v, u)
                            available_core_edges.discard((a, b))
                            core_adj[a][b] = False
                            core_adj[b][a] = False
                    # Remove S from uncovered_cross
                    uncovered_cross -= set(S)
                else:
                    # No core edge can be covered by a clique of size >= 2 containing i
                    # Pick any c in uncovered_cross
                    c = next(iter(uncovered_cross))
                    cliques.append([i, c])
                    uncovered_cross.discard(c)
                    # No core edges removed
                    
        # If we broke out of the loop due to deadline, we must handle remaining I vertices
        # But the loop iterates through all I. If we break, we haven't processed the rest.
        # We must ensure we process all i.
        # The operations per i are small.
        
        # Now cover remaining core edges
        remaining_edges = set(available_core_edges)
        
        # Rebuild core_adj from remaining_edges for the final phase
        core_adj_final = [[False] * k for _ in range(k)]
        for u in range(k):
            core_adj_final[u][u] = False
        for (u, v) in remaining_edges:
            core_adj_final[u][v] = True
            core_adj_final[v][u] = True
            
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: use 2-cliques for all remaining
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Find a maximal clique in the graph defined by remaining_edges
            # Vertices involved in remaining_edges
            vertices_in_remaining = set()
            for (u, v) in remaining_edges:
                vertices_in_remaining.add(u)
                vertices_in_remaining.add(v)
                
            if not vertices_in_remaining:
                break
                
            # Find max clique in the subgraph induced by vertices_in_remaining
            V_list = list(vertices_in_remaining)
            m = len(V_list)
            temp_adj = [[False] * m for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    if i != j:
                        u = V_list[i]
                        v = V_list[j]
                        temp_adj[i][j] = core_adj_final[u][v]
                        
            best = []
            def dfs2(candidate, available_indices):
                nonlocal best
                if not available_indices:
                    if len(candidate) > len(best):
                        best = candidate[:]
                    return
                for idx, i in enumerate(available_indices):
                    new_candidate = candidate + [V_list[i]]
                    new_available = [j for j in available_indices[idx+1:] if temp_adj[i][j]]
                    dfs2(new_candidate, new_available)
                    
            dfs2([], list(range(m)))
            
            S = [V_list[i] for i in best]
            
            if len(S) < 2:
                # If S has 0 or 1 vertices, it covers no edges.
                # Pick an arbitrary edge from remaining_edges
                if remaining_edges:
                    u, v = next(iter(remaining_edges))
                    S = [u, v]
                else:
                    break
            
            if len(S) >= 2:
                cliques.append(S)
                
                # Remove edges within S from remaining_edges
                S_list = list(S)
                for idx_u in range(len(S_list)):
                    for idx_v in range(idx_u + 1, len(S_list)):
                        u = S_list[idx_u]
                        v = S_list[idx_v]
                        a, b = (u, v) if u < v else (v, u)
                        remaining_edges.discard((a, b))
        
        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve_safe(*spec, deadline))

    return results
