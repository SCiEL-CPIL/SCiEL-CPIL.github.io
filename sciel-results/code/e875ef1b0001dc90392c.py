def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        # Build neighbors for I vertices
        cross = {i: set() for i in I}
        for (a, b) in edges:
            if a < k and b >= k:
                cross[b].add(a)
            elif a >= k and b < k:
                cross[a].add(b)
        
        # Step 1: Greedy Set Cover for Cross Edges
        # Universe: All cross edges.
        # Candidate Sets: For each i, we can pick a clique {i} U S where S subset of N(i).
        # The edges covered are {(i, v) for v in S}.
        # To maximize coverage, we pick S = N(i).
        # This results in one mixed clique per i with N(i) != empty.
        
        mixed_cliques = []
        covered_pure_edges = set()
        
        for i in I:
            N_i = cross[i]
            if len(N_i) > 0:
                clique = [i] + list(N_i)
                mixed_cliques.append(clique)
                # Mark pure edges within N_i as covered
                for u, v in itertools.combinations(N_i, 2):
                    covered_pure_edges.add(frozenset((u, v)))
        
        # Step 2: Cover remaining pure edges
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))
        remaining_edges = all_C_edges - covered_pure_edges
        
        pure_cliques = []
        rem = set(remaining_edges)
        
        # Get vertices involved in remaining edges
        active_vertices = set()
        for e in rem:
            active_vertices.update(e)
        
        # Greedy Clique Cover on remaining edges
        # We want to cover all edges in 'rem' with minimum number of cliques.
        # Heuristic: Repeatedly find a maximal clique in the graph defined by 'rem',
        # preferably one that covers many edges.
        
        while rem:
            # Build adjacency for current remaining graph
            # Vertices are from active_vertices
            # Edge exists if frozenset(u,v) in rem
            
            # Find a good starting vertex: highest degree in rem
            degs = {v: 0 for v in active_vertices}
            for e in rem:
                u, v = tuple(e)
                degs[u] += 1
                degs[v] += 1
            
            if not active_vertices:
                break
                
            # Pick vertex with max degree
            start_v = max(active_vertices, key=lambda x: degs[x])
            
            # Greedy expansion to find a maximal clique containing start_v
            clique = [start_v]
            clique_set = {start_v}
            
            # Candidates: neighbors of all in clique_set in the remaining graph
            # Initially, neighbors of start_v
            candidates = set()
            for e in rem:
                if start_v in e:
                    other = e.difference({start_v}).pop()
                    candidates.add(other)
            
            # Sort candidates by degree in the induced subgraph of candidates?
            # Or just pick one with max degree in rem
            while candidates:
                # Pick candidate with max degree in the current candidate set
                # (This helps find a large clique)
                best_cand = None
                max_deg = -1
                for v in candidates:
                    # Count neighbors in candidates
                    deg = 0
                    for e in rem:
                        if v in e:
                            other = e.difference({v}).pop()
                            if other in candidates:
                                deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        best_cand = v
                
                if best_cand is None:
                    break
                
                clique.append(best_cand)
                clique_set.add(best_cand)
                
                # Update candidates: must be connected to best_cand and all in clique_set
                new_candidates = set()
                for v in candidates:
                    if v == best_cand:
                        continue
                    # Check if v connected to best_cand
                    connected_to_best = False
                    for e in rem:
                        if best_cand in e:
                            other = e.difference({best_cand}).pop()
                            if other == v:
                                connected_to_best = True
                                break
                    if connected_to_best:
                        # Check if v connected to all in clique_set (already true for previous, check new)
                        # Actually, since we maintain candidates as common neighbors,
                        # we just need to filter by best_cand
                        new_candidates.add(v)
                
                candidates = new_candidates
            
            # Add clique to pure_cliques
            if len(clique) >= 2:
                pure_cliques.append(clique)
                
            # Remove edges covered by this clique
            clique_set_final = set(clique)
            edges_to_remove = set()
            for e in rem:
                if e.issubset(clique_set_final):
                    edges_to_remove.add(e)
            
            for e in edges_to_remove:
                rem.discard(e)
            
            # Update active_vertices? Not strictly necessary if we iterate rem,
            # but degs calculation uses active_vertices.
            # We can just recalculate active_vertices from rem next time.
            # Or just let degs be calculated on the fly in the loop.
            # The 'degs' calculation above iterates over rem, so it's fine.
            # But 'active_vertices' is static. We should update it or just ignore it for selection
            # and rely on rem.
            # Actually, if a vertex has no edges left, it won't be picked because deg=0.
            # But 'max' might pick a vertex with deg 0 if all are 0?
            # If rem is not empty, some vertex has deg > 0.
            
            # Optimization: If rem is small, this is fast.
            
        result = mixed_cliques + pure_cliques
        
        # Ensure types
        return [[int(x) for x in clique] for clique in result]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
