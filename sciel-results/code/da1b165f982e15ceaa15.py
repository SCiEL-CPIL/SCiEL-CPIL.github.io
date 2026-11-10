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
        
        # Build adjacency and edge sets
        # cross_edges: set of frozenset((i, c))
        # c_edges: set of frozenset((c1, c2))
        cross_edges = set()
        c_edges = set()
        for (a, b) in edges:
            if a < k and b >= k:
                cross_edges.add(frozenset((a, b)))
            elif a >= k and b < k:
                cross_edges.add(frozenset((a, b)))
            elif a < k and b < k:
                c_edges.add(frozenset((a, b)))
            # a>=k and b>=k is impossible as I is independent

        # Uncovered sets
        uncovered_cross = set(cross_edges)
        covered_c = set() # c_edges already covered by mixed cliques
        
        result_cliques = []
        
        # Greedy Set Cover for cross edges
        while uncovered_cross:
            best_clique = None
            best_count = 0
            best_covered_c = set()
            
            for i in I:
                # Find neighbors of i in uncovered_cross
                # Neighbors are c in C such that frozenset((i, c)) in uncovered_cross
                candidates = []
                for c in C:
                    if frozenset((i, c)) in uncovered_cross:
                        candidates.append(c)
                
                if not candidates:
                    continue
                
                # Greedily build S from candidates
                # S must satisfy: for all u, v in S, frozenset((u, v)) not in covered_c
                S = []
                S_set = set()
                # Sort candidates? Maybe by degree in remaining c-edges? 
                # For simplicity, just iterate.
                for c in candidates:
                    # Check if c can be added to S
                    can_add = True
                    for v in S:
                        if frozenset((c, v)) in covered_c:
                            can_add = False
                            break
                    if can_add:
                        S.append(c)
                        S_set.add(c)
                
                if len(S) > best_count:
                    best_count = len(S)
                    best_clique = [i] + S
                    # Calculate which c-edges are covered by this S
                    best_covered_c = set(frozenset((u, v)) for u, v in itertools.combinations(S, 2))
            
            if best_clique is None:
                # Should not happen if uncovered_cross is not empty
                # This would imply a cross edge exists but no i can cover it?
                # Or all candidates led to empty S?
                # If uncovered_cross is not empty, there is some (i, c).
                # Then for that i, c is a candidate. S will at least contain c.
                # Unless... covered_c blocks it? No, S starts empty.
                # So S will have at least 1 element.
                break
            
            result_cliques.append(best_clique)
            
            # Update uncovered_cross
            for c in best_clique[1:]:
                uncovered_cross.discard(frozenset((best_clique[0], c)))
            
            # Update covered_c
            covered_c.update(best_covered_c)
        
        # Cover remaining C-edges with pure cliques
        remaining_c_edges = c_edges - covered_c
        
        pure_cliques = []
        rem = set(remaining_c_edges)
        C_list = list(range(k))
        
        while rem:
            # Find a maximal clique in the graph defined by rem
            # Greedy: pick a vertex with max degree in rem, then expand
            # Or just pick first, expand.
            
            best_clique = []
            best_size = 0
            
            # Try starting from each vertex? That's O(k * E). 
            # Better: Pick vertex with max degree in rem as start.
            # Calculate degrees in rem
            deg = {v: 0 for v in C_list}
            for e in rem:
                u, v = tuple(e)
                deg[u] += 1
                deg[v] += 1
            
            # Sort vertices by degree descending
            sorted_C = sorted(C_list, key=lambda x: deg[x], reverse=True)
            
            for start in sorted_C:
                # Greedy expansion
                clique = [start]
                clique_set = {start}
                
                # Candidates must be connected to all in clique via rem
                candidates = [v for v in sorted_C if v != start and frozenset((start, v)) in rem]
                
                while candidates:
                    # Pick candidate with max degree in rem within candidates
                    max_deg = -1
                    best_cand = None
                    for v in candidates:
                        # Count neighbors in candidates
                        d = 0
                        for w in candidates:
                            if w != v and frozenset((v, w)) in rem:
                                d += 1
                        if d > max_deg:
                            max_deg = d
                            best_cand = v
                    
                    if best_cand is None:
                        break
                        
                    clique.append(best_cand)
                    clique_set.add(best_cand)
                    
                    # Update candidates
                    new_candidates = []
                    for w in candidates:
                        if w == best_cand:
                            continue
                        if all(frozenset((w, u)) in rem for u in clique_set):
                            new_candidates.append(w)
                    candidates = new_candidates
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = list(clique)
            
            if best_size < 2:
                # If we can't form a clique of size 2, but rem is not empty,
                # it means rem has edges but no connected components?
                # Wait, if rem has an edge (u, v), then u and v form a clique of size 2.
                # So best_size should be at least 2.
                # Unless my greedy failed to find it?
                # If rem is not empty, there is at least one edge.
                # The loop should find at least a pair.
                break
                
            pure_cliques.append(best_clique)
            
            # Remove edges covered by best_clique
            clique_set = set(best_clique)
            for e in list(rem):
                if e.issubset(clique_set):
                    rem.discard(e)
        
        result = result_cliques + pure_cliques
        # Ensure types
        return [[int(x) for x in clique] for clique in result]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
