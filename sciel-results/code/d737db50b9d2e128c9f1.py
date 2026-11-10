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
        
        # Initial partitions: singletons
        partitions = {i: [[c] for c in cross[i]] for i in I}
        used_C_edges = set()
        
        # Greedy merging
        changed = True
        while changed:
            changed = False
            for i in I:
                parts = partitions[i]
                if len(parts) < 2:
                    continue
                # Try to merge any pair
                merged = False
                # Sort parts by size descending to prefer merging larger sets?
                # Or just iterate all pairs.
                for idx_a in range(len(parts)):
                    if merged:
                        break
                    for idx_b in range(idx_a + 1, len(parts)):
                        A = parts[idx_a]
                        B = parts[idx_b]
                        # Check edges between A and B
                        conflict = False
                        for u in A:
                            for v in B:
                                e = frozenset((u, v))
                                if e in used_C_edges:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if not conflict:
                            # Merge
                            new_set = A + B
                            # Update partitions
                            # Remove A and B, add new_set
                            # Careful with indices if we modify list in place
                            # Create new list
                            new_parts = []
                            for idx in range(len(parts)):
                                if idx != idx_a and idx != idx_b:
                                    new_parts.append(parts[idx])
                            new_parts.append(new_set)
                            partitions[i] = new_parts
                            
                            # Add all edges in new_set to used_C_edges
                            for u, v in itertools.combinations(new_set, 2):
                                used_C_edges.add(frozenset((u, v)))
                            
                            changed = True
                            merged = True
                            break # Restart pair loop for this i? 
                            # Actually, we modified partitions[i], so we should break out of pair loops
                            # and let the while loop restart or just continue to next i
                            # Breaking pair loop is enough, but we need to ensure we don't use stale 'parts'
                            # Since we break inner loop, we go to next i or end of i loop
                            # The 'changed' flag ensures we loop again
                            pass

        # Construct mixed cliques
        mixed_cliques = []
        for i in I:
            for S in partitions[i]:
                if len(S) >= 1:
                    mixed_cliques.append([i] + S)
                # Note: if S is empty, we don't add a clique, but N(i) is covered by partition so no empty S unless N(i) was empty.
                # If N(i) is empty, no cross edges, no mixed clique needed.
        
        # Cover remaining C-edges
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))
        remaining_C_edges = all_C_edges - used_C_edges
        
        pure_cliques = []
        rem = set(remaining_C_edges)
        C_list = list(range(k))
        
        while rem:
            # Find a clique in the graph defined by rem
            # Greedy: pick a vertex, expand
            # Since it's a subgraph of K_k, any subset is a clique if all edges present.
            # We want a maximal clique in the "edge graph".
            # Actually, we want a set of vertices V' such that all edges in V' are in rem.
            # This is a clique in the graph G_rem = (C, rem).
            
            best_clique = []
            best_size = 0
            
            # Try starting from each vertex
            for start in C_list:
                # Greedy expansion
                clique = [start]
                clique_set = {start}
                # Candidates must be connected to all in clique via rem
                candidates = [v for v in C_list if v != start]
                
                # Filter candidates that have edges to all in clique
                # Initially just need edge to start
                candidates = [v for v in candidates if frozenset((start, v)) in rem]
                
                while candidates:
                    # Pick candidate with max degree in rem within candidates?
                    # Or just first?
                    # Let's pick the one with max degree in the induced subgraph of candidates
                    # For simplicity, pick first, but maybe better to pick max degree
                    max_deg = -1
                    best_cand = None
                    for v in candidates:
                        deg = sum(1 for w in candidates if w != v and frozenset((v, w)) in rem)
                        if deg > max_deg:
                            max_deg = deg
                            best_cand = v
                    
                    if best_cand is None:
                        break
                        
                    clique.append(best_cand)
                    clique_set.add(best_cand)
                    
                    # Update candidates
                    candidates = [w for w in candidates if w != best_cand and all(frozenset((w, u)) in rem for u in clique_set)]
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = list(clique)
            
            if best_size < 2:
                # Should not happen if rem is not empty, but just in case
                break
                
            pure_cliques.append(best_clique)
            
            # Remove edges covered by best_clique
            clique_set = set(best_clique)
            for e in list(rem):
                if e.issubset(clique_set):
                    rem.discard(e)
        
        result = mixed_cliques + pure_cliques
        # Ensure types
        return [[int(x) for x in clique] for clique in result]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
