    def solve_one(n, k, p, seed, deadline):
        edges = instance(n, k, p, seed)
        E = set()
        for a, b in edges:
            E.add((a, b))
        C = set(range(k))
        I = set(range(k, n))
        
        adj = {c: set() for c in C}
        E_cross = set()
        for (a, b) in E:
            if a < k and b >= k:
                adj[a].add(b)
                E_cross.add((a, b))
            elif b < k and a >= k:
                adj[b].add(a)
                E_cross.add((b, a))
        
        cliques = []
        remaining_edges = set(E_cross)
        active_c = set(C)
        
        while remaining_edges:
            if time.time() > deadline:
                break
            best_c = -1
            best_s = set()
            best_count = 0
            for c in active_c:
                s = set()
                for i in adj[c]:
                    if (c, i) in remaining_edges:
                        s.add(i)
                if len(s) > best_count:
                    best_count = len(s)
                    best_c = c
                    best_s = s
            
            if best_c == -1:
                break
                
            clique = [best_c] + list(best_s)
            cliques.append(clique)
            for i in best_s:
                remaining_edges.discard((best_c, i))
            active_c.discard(best_c)
            
        # Local improvement: Merge cliques
        # A clique is a list of vertices.
        # We can merge clique A and B if A U B is a clique.
        # Since C is a clique and I is independent, A U B is a clique iff
        # all I vertices in A U B are connected to all C vertices in A U B.
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            m = len(cliques)
            for i in range(m):
                if time.time() > deadline:
                    break
                for j in range(i + 1, m):
                    if time.time() > deadline:
                        break
                    A = cliques[i]
                    B = cliques[j]
                    A_set = set(A)
                    B_set = set(B)
                    Union = A_set | B_set
                    
                    # Check if Union is a clique
                    # Extract C and I parts
                    C_union = {v for v in Union if v < k}
                    I_union = {v for v in Union if v >= k}
                    
                    is_clique = True
                    # Check I to C connections
                    for c in C_union:
                        for i in I_union:
                            if (c, i) not in E: # Or adj[c] doesn't contain i
                                is_clique = False
                                break
                        if not is_clique: break
                    # Check C to C (always true)
                    # Check I to I (always false, but I is independent so no edges needed)
                    
                    if is_clique:
                        # Merge
                        new_clique = list(Union)
                        # Remove i, j
                        cliques = [c for idx, c in enumerate(cliques) if idx != i and idx != j]
                        cliques.append(new_clique)
                        improved = True
                        break
                if improved:
                    break
                    
        # Add the main clique C
        cliques.append(list(C))
        
        # Verification
        cover = {}
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    key = (min(cl[a], cl[b]), max(cl[a], cl[b]))
                    cover[key] = cover.get(key, 0) + 1
        for key in cover:
            if cover[key] != 1:
                raise RuntimeError("overlap/miss")
        if set(cover.keys()) != E:
            raise RuntimeError("edge set mismatch")
            
        return cliques
