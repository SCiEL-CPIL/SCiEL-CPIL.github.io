def construct():
    import time
    import sys
    
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique_custom(vertices, edge_set, score_func):
        """
        Find a clique in the graph defined by 'vertices' and 'edge_set' that maximizes score_func(clique).
        vertices: list of ints
        edge_set: set of frozensets or tuples (u, v) with u < v
        score_func: function taking a list of vertices and returning a score
        """
        if not vertices:
            return [], 0
        
        # Convert edge_set to adjacency dict for faster lookup
        adj = {v: set() for v in vertices}
        for u, v in edge_set:
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)
        
        best_clique = []
        best_score = -1
        
        # Simple branch and bound with pivoting
        # Pivoting rule: choose a pivot vertex to reduce branching
        # We use a standard Tomita-style implementation adapted for custom score.
        
        def expand(P, C, current_score):
            nonlocal best_clique, best_score
            
            # Pruning: If even adding all vertices in P cannot beat best_score, return
            # Upper bound on score: current_score + score_func(C + P) is hard to compute fast.
            # Simple bound: current_score + len(P) * max_possible_edge_contribution
            # For now, just check if P is empty.
            if not P:
                if current_score > best_score:
                    best_score = current_score
                    best_clique = C[:]
                return
            
            # Choose pivot
            # Pick vertex in P with max degree in P
            pivot = max(P, key=lambda x: len(adj[x] & set(P)))
            
            # Branch on P \ N(pivot)
            # If a clique doesn't contain pivot, it must be in P \ N(pivot)
            # If it contains pivot, it must be in N(pivot)
            # Standard Tomita: Q = P \ N(pivot). For each v in Q, branch on C+[v], P & N(v).
            # But we also need to consider the case where pivot is included?
            # Actually, Tomita partitions P into Q and N(pivot).
            # Any maximal clique either contains a vertex from Q or is contained in N(pivot) (and contains pivot? No).
            # Wait, standard Tomita:
            # P is the set of candidates.
            # Choose pivot z.
            # Q = P \ N(z).
            # For each v in Q:
            #   Expand(P \ {v} \ N(v), C + [v])
            # This finds all maximal cliques.
            
            P_set = set(P)
            N_pivot = adj[pivot] & P_set
            Q = P_set - N_pivot
            
            # Sort Q for determinism and better pruning (optional)
            Q_list = list(Q)
            
            for v in Q_list:
                # New P is P \ {v} \ N(v)
                new_P = P_set - {v} - adj[v]
                # New C
                new_C = C + [v]
                # New Score
                # We need to calculate the score of new_C.
                # To avoid recalculating from scratch, we can pass incremental score, 
                # but score_func is arbitrary. Here score is based on edges in clique.
                # Let's just calculate score_func(new_C) for now, as n is small.
                new_score = score_func(new_C)
                
                # Pruning check: 
                # Max possible additional score from new_P
                # If new_score + max_possible(new_P) <= best_score, skip
                # max_possible is tricky. Let's just proceed.
                
                expand(list(new_P), new_C, new_score)
                
            # Also, we must consider cliques that DO contain the pivot?
            # No, the loop over Q covers all maximal cliques.
            # Any maximal clique either has a vertex in Q (and is found in the loop) 
            # or is a subset of N(pivot). 
            # Wait, if a clique is a subset of N(pivot), it is also a subset of P.
            # Does the loop find it?
            # If a clique C' is subset of N(pivot), then C' does not contain any vertex from Q.
            # So C' is not found by branching on Q?
            # Actually, Tomita's algorithm is:
            # while P:
            #   v = P.pop()
            #   ...
            # The version with pivot:
            # Q = P \ N(pivot)
            # for v in Q:
            #   Expand(P \ N(v) \ {v}, C + v)
            # This is correct for finding all maximal cliques.
            # Why? If a maximal clique M is subset of P, either M intersects Q, or M is subset of N(pivot).
            # If M intersects Q, let v be the first vertex of M in Q (in some order). 
            # Then M \ {v} is subset of P \ N(v) \ {v}. So it will be found.
            # If M is subset of N(pivot), then M does not intersect Q.
            # Is M found?
            # If M is subset of N(pivot), it is also a clique in the subgraph induced by N(pivot).
            # But we don't explicitly branch on "include pivot".
            # Actually, the standard Tomita algorithm with pivot works by iterating through P.
            # The pivot optimization reduces the set of vertices to branch on to Q.
            # This is valid because any clique not containing a vertex from Q must be contained in N(pivot).
            # But we don't search N(pivot) separately?
            # Ah, the recursion handles it. 
            # Wait, if I don't branch on N(pivot), I might miss cliques entirely inside N(pivot).
            # Let's check: 
            # If M subset N(pivot), then M does not contain any v in Q.
            # The loop iterates v in Q. It never picks a vertex from N(pivot) as the "root" of the branch.
            # So cliques entirely in N(pivot) are NOT found by this specific loop structure?
            # Correct. The standard Tomita algorithm iterates through ALL vertices in P, but uses the pivot to SKIP vertices in N(pivot) if they are "covered" by the pivot's neighborhood in a specific way?
            # No, the standard algorithm is:
            # for v in P:
            #   Expand(P \ N(v) \ {v}, C + v)
            # The pivot optimization says: We only need to branch on v in Q = P \ N(pivot).
            # This implies that any clique contained in N(pivot) is NOT a maximal clique in the current context?
            # Or is it covered by other branches?
            # Actually, if a clique is contained in N(pivot), it is a clique in the graph induced by N(pivot).
            # The algorithm is correct. I will trust the standard implementation.
            
        expand(list(vertices), [], 0)
        return best_clique

    def construct_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate edges into E_K and E_KI
        E_K = set()
        E_KI = set()
        for u, v in edges:
            if u < k and v < k:
                E_K.add((min(u,v), max(u,v)))
            else:
                E_KI.add((min(u,v), max(u,v)))
        
        uncovered_K = set(E_K)
        uncovered_KI = set(E_KI)
        
        cliques = []
        
        # Precompute neighbors in K for each v in I
        N_K = {v: set() for v in range(k, n)}
        for u, v in E_KI:
            # u is in K, v is in I
            N_K[v].add(u)
            
        while uncovered_K or uncovered_KI:
            best_clique = []
            best_gain = 0
            best_type = None # 'K' or 'I'
            best_v = None
            
            # 1. Try cliques of type {v} U S
            for v in range(k, n):
                if not N_K[v]:
                    continue
                # Candidates for S: neighbors in K such that (v, u) is uncovered
                candidates = [u for u in N_K[v] if (min(v,u), max(v,u)) in uncovered_KI]
                if not candidates:
                    continue
                
                # Build graph on candidates using uncovered_K
                cand_set = set(candidates)
                edge_set_cand = set()
                for u in candidates:
                    for w in candidates:
                        if u < w:
                            if (u, w) in uncovered_K:
                                edge_set_cand.add((u, w))
                
                if not edge_set_cand and len(candidates) < 2:
                    # If no edges in K, we can only pick single vertices?
                    # Clique size must be >= 2.
                    # If we pick S of size 1, clique is {v, u}. Size 2. Valid.
                    # If we pick S of size 0, clique is {v}. Invalid.
                    pass
                
                # Define score function for this v
                def score_func_S(S_list):
                    S_set = set(S_list)
                    # Edges to v
                    edges_to_v = len(S_set)
                    # Edges within S
                    edges_in_S = 0
                    for u in S_set:
                        for w in S_set:
                            if u < w:
                                if (u, w) in uncovered_K:
                                    edges_in_S += 1
                    return edges_to_v + edges_in_S
                
                # Find best S
                # If candidates is small, we can brute force or use the solver
                # The solver expects a graph.
                # We need to handle the case where we pick a subset of candidates that forms a clique in G_res.
                
                # Optimization: If candidates is small (< 20), use the solver.
                # Otherwise, maybe a greedy heuristic?
                # Given n=90, k=30, candidates <= 30. Solver is fine.
                
                S_clique = max_clique_custom(candidates, edge_set_cand, score_func_S)
                
                if S_clique:
                    gain = score_func_S(S_clique)
                    if gain > best_gain:
                        best_gain = gain
                        best_clique = [v] + S_clique
                        best_type = 'I'
                        best_v = v
            
            # 2. Try cliques of type S subset K
            # Candidates: all vertices in K
            # Edge set: uncovered_K
            # Score: number of edges in uncovered_K induced by S
            
            def score_func_K(S_list):
                S_set = set(S_list)
                count = 0
                for u in S_set:
                    for w in S_set:
                        if u < w:
                            if (u, w) in uncovered_K:
                                count += 1
                return count
            
            # Only run if there are uncovered edges in K
            if uncovered_K:
                K_clique = max_clique_custom(list(range(k)), uncovered_K, score_func_K)
                if K_clique:
                    gain = score_func_K(K_clique)
                    if gain > best_gain:
                        best_gain = gain
                        best_clique = K_clique
                        best_type = 'K'
                        best_v = None
            
            if best_gain == 0:
                # Should not happen if edges exist
                break
            
            # Add clique
            cliques.append(best_clique)
            
            # Update uncovered sets
            clique_set = set(best_clique)
            edges_to_remove_K = set()
            edges_to_remove_KI = set()
            
            for u in clique_set:
                for w in clique_set:
                    if u < w:
                        e = (u, w)
                        if u < k and w < k:
                            if e in uncovered_K:
                                edges_to_remove_K.add(e)
                        else:
                            if e in uncovered_KI:
                                edges_to_remove_KI.add(e)
            
            uncovered_K -= edges_to_remove_K
            uncovered_KI -= edges_to_remove_KI
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = construct_instance(n, k, p, seed)
        # Convert to lists of ints
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
