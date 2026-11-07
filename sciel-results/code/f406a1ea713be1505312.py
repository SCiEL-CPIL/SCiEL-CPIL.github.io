def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    DEADLINE = 115  # safety margin before 120s kill (evaluator kills at 120s)

    def _time_up():
        return (time.time() - start_time) > DEADLINE

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Edges within clique part
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def _greedy_clique_cover(adj, active_vertices):
        """
        Greedy minimum clique cover on a graph given by adjacency sets.
        Equivalent to greedy coloring of the complement graph.
        Returns list of cliques (each a list of vertices).
        """
        remaining = set(active_vertices)
        cliques = []
        
        # Precompute degrees for ordering
        while remaining:
            # Build the subgraph induced by remaining
            # We want to find a maximum clique in the remaining graph
            # Greedy heuristic: start with vertex of highest degree, add compatible vertices
            
            # Sort vertices by degree (descending) to prioritize "hub" vertices
            # For clique finding, high degree in original = high "non-clique" in complement
            # Actually, we are coloring the COMPLEMENT.
            # So we want independent sets in complement = cliques in original.
            # Greedy coloring: pick vertex with highest degree in complement?
            # Or just pick any, then find largest independent set in complement?
            
            # Let's use a simple greedy: pick vertex with highest degree in ORIGINAL graph
            # (which means it has fewest non-edges, so it's "central" for cliques).
            # Then greedily add vertices connected to ALL current clique members.
            
            # Calculate degrees within remaining
            degs = {}
            for v in remaining:
                degs[v] = len(adj[v] & remaining)
            
            # Pick best start vertex
            start_v = max(remaining, key=lambda x: degs[x])
            current_clique = [start_v]
            remaining.discard(start_v)
            
            # Candidates: vertices in remaining connected to start_v
            candidates = [v for v in remaining if start_v in adj[v]]
            
            # Greedily add candidates that connect to all current clique members
            # Sort candidates by degree in remaining (descending) for better packing
            candidates.sort(key=lambda x: -degs[x])
            
            for v in candidates:
                if v not in remaining:
                    continue
                # Check if v connects to all in current_clique
                is_connected = True
                for u in current_clique:
                    if u not in adj[v]:
                        is_connected = False
                        break
                if is_connected:
                    current_clique.append(v)
                    remaining.discard(v)
            
            if len(current_clique) >= 2:
                cliques.append(sorted(current_clique))
            # If clique size is 1, we still need to cover the vertex?
            # No, the problem says "clique with at least two vertices".
            # But every EDGE must be covered.
            # If a vertex has no edges to other active vertices, it doesn't need to be in a clique
            # UNLESS it has edges to vertices already covered?
            # Wait, the input to this function is the graph of REMAINING edges.
            # If a vertex is isolated in the remaining graph, it has no remaining edges.
            # So it doesn't need to be covered by a new clique.
            # However, if we discard it, we must ensure its edges are covered.
            # By definition, if it's in `remaining` and isolated, it has no edges in the
            # subgraph induced by `remaining`. But it might have edges to vertices NOT in remaining?
            # No, `active_vertices` is the set of vertices involved in remaining edges.
            # So if v is in remaining and deg(v)=0 in remaining, it has no edges to cover.
            # So we can just discard it.
            
            # Correction: The loop above adds vertices to clique.
            # If current_clique ends up size 1, we discard it.
            # But we must make sure we don't lose edges.
            # If start_v has no neighbors in remaining, it has no edges to cover.
            
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # 1. Identify Cross Edges and C-C Edges
        cc_edges = set()
        cross_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                cc_edges.add((u, v) if u < v else (v, u))
            elif (u < k and v >= k) or (u >= k and v < k):
                cross_edges.add((u, v) if u < v else (v, u))
        
        # 2. Determine neighbors in C for each I-vertex
        I_verts = list(range(k, n))
        N_C = {}
        d_v = {}
        for v in I_verts:
            neigh = set()
            for u in range(k):
                if (u, v) in cross_edges or (v, u) in cross_edges:
                    neigh.add(u)
            N_C[v] = neigh
            d_v[v] = len(neigh)
            
        # 3. Precompute C-C edges within N_C[v]
        # E_N[v] = set of C-C edges that are "inside" the neighborhood of v
        # These edges can be covered by the clique {v} U N_C[v]
        E_N = {}
        for v in I_verts:
            neigh = list(N_C[v])
            s = set()
            for a, b in combinations(neigh, 2):
                e = (a, b) if a < b else (b, a)
                if e in cc_edges:
                    s.add(e)
            E_N[v] = s

        # 4. Search for optimal set S of I-vertices to form "Big Cliques"
        # Big Clique for v: {v} U N_C[v]
        # This covers all cross edges incident to v, and all C-C edges within N_C[v].
        # Cost of Big Clique: 1 clique.
        # Cost of NOT Big (for v): d_v[v] cliques of size 2 {v, u}.
        
        # We need to cover remaining C-C edges with C-only cliques.
        
        def evaluate(S):
            """
            S: set of I-vertices using Big Cliques.
            Returns: (total_cliques, list_of_cliques)
            """
            cliques = []
            
            # Add Big Cliques
            for v in S:
                if d_v[v] >= 2: # Only use big clique if it saves at least 1 clique (d_v - 1 > 0 => d_v >= 2)
                    cliques.append(sorted([v] + list(N_C[v])))
                else:
                    # If d_v < 2, big clique is not beneficial or impossible (size 1 or 2)
                    # If d_v == 1, Big Clique is size 2, same as individual.
                    # If d_v == 0, no cross edges.
                    # We handle these in the "individual" part to avoid double counting or logic errors.
                    # Actually, if v is in S but d_v < 2, we shouldn't have put it there ideally.
                    # But for evaluation, let's just treat it as individual if it's not a "good" big clique.
                    # Better: only allow S to contain v if d_v[v] >= 2.
                    pass 
            
            # Add Individual Cliques for I-vertices NOT in S (or in S but d_v < 2)
            for v in I_verts:
                if v not in S or d_v[v] < 2:
                    for u in N_C[v]:
                        cliques.append(sorted([u, v]))
            
            # Determine remaining C-C edges
            consumed_cc = set()
            for v in S:
                if d_v[v] >= 2:
                    consumed_cc |= E_N[v]
            
            remaining_cc = cc_edges - consumed_cc
            
            # Cover remaining C-C edges
            if remaining_cc:
                # Build adjacency for remaining C-C graph
                rem_adj = {u: set() for u in range(k)}
                active_c = set()
                for u, v in remaining_cc:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
                    active_c.add(u)
                    active_c.add(v)
                
                rem_cliques = _greedy_clique_cover(rem_adj, active_c)
                cliques.extend(rem_cliques)
            
            return len(cliques), cliques

        # Initial Candidates for S
        candidates = []
        
        # 1. All vertices with d_v >= 2
        S_all = frozenset(v for v in I_verts if d_v[v] >= 2)
        candidates.append(S_all)
        
        # 2. Empty set
        candidates.append(frozenset())
        
        # 3. Greedy construction: Add vertex with best "gain"
        # Gain = (d_v - 1) - (cost increase in C-C cover)
        # Approximate gain = d_v - 1 - (number of unique new C-C edges consumed)
        # This is complex. Let's just try a few heuristics.
        
        # Sort I-vertices by d_v descending
        sorted_I = sorted(I_verts, key=lambda v: -d_v[v])
        
        S_greedy = frozenset()
        for v in sorted_I:
            if d_v[v] >= 2:
                S_greedy = S_greedy | frozenset([v])
        candidates.append(S_greedy)
        
        # 4. Random samples
        for _ in range(10):
            if _time_up(): break
            S_rand = frozenset()
            for v in I_verts:
                if d_v[v] >= 2:
                    if random.random() < 0.5:
                        S_rand = S_rand | frozenset([v])
            candidates.append(S_rand)

        # Evaluate candidates
        best_val = float('inf')
        best_S = frozenset()
        best_cliques = []
        
        for S in candidates:
            if _time_up(): break
            val, cliques = evaluate(S)
            if val < best_val:
                best_val = val
                best_S = S
                best_cliques = cliques
        
        # Local Search: 1-flip moves
        # Try adding/removing vertices from S
        improved = True
        while improved and not _time_up():
            improved = False
            # Try adding vertices
            for v in I_verts:
                if _time_up(): break
                if v in best_S or d_v[v] < 2:
                    continue
                S_new = best_S | frozenset([v])
                val_new, cliques_new = evaluate(S_new)
                if val_new < best_val:
                    best_val = val_new
                    best_S = S_new
                    best_cliques = cliques
                    improved = True
            
            # Try removing vertices
            for v in best_S:
                if _time_up(): break
                S_new = best_S - frozenset([v])
                val_new, cliques_new = evaluate(S_new)
                if val_new < best_val:
                    best_val = val_new
                    best_S = S_new
                    best_cliques = cliques
                    improved = True

        # Final Polish: Try 2-flips? Or just return best found.
        # Given time constraints, 1-flip local search is a good balance.
        
        return best_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for spec in specs:
        if _time_up():
            # Fallback: Trivial solution
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for u, v in edges:
                if (u < k and v >= k) or (u >= k and v < k):
                    cliques.append(sorted([u, v]))
            results.append(cliques)
            continue
        
        results.append(solve_one(*spec))
    
    # Ensure we have 3 results
    while len(results) < 3:
        spec = specs[len(results)]
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        for u, v in edges:
            if (u < k and v >= k) or (u >= k and v < k):
                cliques.append(sorted([u, v]))
        results.append(cliques)

    # Validation and Type Conversion
    final_results = []
    for res in results:
        # Ensure types are correct
        valid_cliques = []
        for c in res:
            if isinstance(c, (list, tuple)):
                # Ensure all elements are ints
                if all(isinstance(x, int) for x in c):
                    # Ensure distinct
                    if len(set(c)) == len(c) and len(c) >= 2:
                        valid_cliques.append(list(c))
        final_results.append(valid_cliques)
        
    return final_results
