def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        # Build adjacency for I vertices to C vertices
        cross = {i: set() for i in I}
        C_edges_set = set()
        
        for (a, b) in edges:
            if a < k and b < k:
                C_edges_set.add(frozenset((a, b)))
            elif a >= k and b >= k:
                pass
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].add(c)
                
        uncovered_cross = set()
        for i in I:
            for c in cross[i]:
                uncovered_cross.add(frozenset((i, c)))
                
        used_C_edges = set()
        cliques = []
        
        # Greedy Set Cover for Cross-Edges
        while uncovered_cross:
            best_i = None
            best_S = None
            best_count = 0
            
            for i in I:
                S = set()
                for c in cross[i]:
                    if frozenset((i, c)) in uncovered_cross:
                        S.add(c)
                        
                if len(S) > best_count:
                    best_count = len(S)
                    best_i = i
                    best_S = S
                    
            if best_i is None or best_count == 0:
                break
                
            clique_vertices = [best_i] + list(best_S)
            cliques.append(clique_vertices)
            
            for c in best_S:
                uncovered_cross.discard(frozenset((best_i, c)))
                
            S_list = list(best_S)
            for idx_a in range(len(S_list)):
                for idx_b in range(idx_a + 1, len(S_list)):
                    used_C_edges.add(frozenset((S_list[idx_a], S_list[idx_b])))
                    
        remaining_C_edges = C_edges_set - used_C_edges
        
        rem_edges = set(remaining_C_edges)
        while rem_edges:
            degrees = {v: 0 for v in C}
            for e in rem_edges:
                v1, v2 = tuple(e)
                degrees[v1] += 1
                degrees[v2] += 1
                
            if not degrees or max(degrees.values()) == 0:
                break
                
            start_v = max(degrees, key=degrees.get)
            clique = {start_v}
            
            changed = True
            while changed:
                changed = False
                candidates = []
                for v in C:
                    if v in clique:
                        continue
                    is_connected = True
                    for u in clique:
                        if frozenset((v, u)) not in rem_edges:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(v)
                
                if candidates:
                    best_cand = max(candidates, key=lambda v: degrees.get(v, 0))
                    clique.add(best_cand)
                    changed = True
            
            cliques.append(list(clique))
            
            clique_list = list(clique)
            to_remove = []
            for e in rem_edges:
                v1, v2 = tuple(e)
                if v1 in clique and v2 in clique:
                    to_remove.append(e)
            for e in to_remove:
                rem_edges.discard(e)
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
