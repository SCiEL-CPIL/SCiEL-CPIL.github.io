def construct():
    import time
    start_time = time.time()
    
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() - start_time > 110:
            # Return what we have, pad with empty lists if necessary? 
            # No, must return 3 lists. 
            # If we fail, we should return a valid but suboptimal solution for the rest.
            # The seed solution is valid.
            # Let's just break and fill with seed.
            pass
            
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cc_edges = set()
        cross_edges = set()
        for u, v in edges:
            if u < k and v < k:
                cc_edges.add((u, v))
            else:
                cross_edges.add((u, v))
                
        remaining_cc = set(cc_edges)
        remaining_cross = set(cross_edges)
        
        cliques = []
        
        N_C = [set() for _ in range(n)]
        for i in range(k, n):
            for c in range(k):
                if c in adj[i]:
                    N_C[i].add(c)
                    
        while remaining_cc and (time.time() - start_time < 115):
            best_size = -1
            best_i = None
            best_S = None
            
            # To speed up, we can limit the number of i's we check or use a heuristic
            # But n is small.
            
            for i in range(k, n):
                if not N_C[i]: continue
                
                # Find max clique in remaining_cc induced by N_C[i]
                # Greedy
                candidates = list(N_C[i])
                # Sort by degree in remaining_cc
                def deg(v):
                    cnt = 0
                    for u in candidates:
                        if u != v and (min(u,v), max(u,v)) in remaining_cc:
                            cnt += 1
                    return cnt
                
                candidates.sort(key=deg, reverse=True)
                
                S = set()
                for v in candidates:
                    ok = True
                    for u in S:
                        if (min(u,v), max(u,v)) not in remaining_cc:
                            ok = False
                            break
                    if ok:
                        S.add(v)
                        
                if len(S) > best_size:
                    best_size = len(S)
                    best_i = i
                    best_S = S
                    if best_size >= k: # Can't get better
                        break
                        
            if best_S is None or len(best_S) < 2:
                break
                
            cliques.append([best_i] + list(best_S))
            
            for u in best_S:
                for v in best_S:
                    if u < v:
                        remaining_cc.discard((u, v))
                remaining_cross.discard((min(best_i, u), max(best_i, u)))

        for u, v in remaining_cc:
            cliques.append([u, v])
            
        for u, v in remaining_cross:
            cliques.append([u, v])
            
        results.append(cliques)
        
    # If we didn't finish all, fill with seed
    while len(results) < 3:
        n, k, p, seed = specs[len(results)]
        edges = instance(n, k, p, seed)
        results.append([list(e) for e in edges])
        
    return results
