def construct():
    from random import Random
    import time
    
    start_time = time.time()
    DEATHLINE = 120
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_greedy(n, k, edges):
        edge_set = set((min(u,v), max(u,v)) for u, v in edges)
        
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        C = set(range(k))
        I = list(range(k, n))
        
        uncovered = set(edge_set)
        cliques = []
        
        N = {v: adj[v] & C for v in I}
        
        while uncovered:
            if time.time() - start_time > DEATHLINE - 5:
                break
            
            best_clique = None
            best_count = 0
            
            # Try C
            c_edges = set()
            c_list = sorted(C)
            for i in range(len(c_list)):
                for j in range(i+1, len(c_list)):
                    e = (c_list[i], c_list[j])
                    if e in uncovered:
                        c_edges.add(e)
            if len(c_edges) > best_count:
                best_count = len(c_edges)
                best_clique = c_list
            
            # Try {v} ∪ N(v) for each v in I
            for v in I:
                Nv = N[v]
                if not Nv:
                    continue
                clique = [v] + sorted(Nv)
                count = 0
                all_uncovered = True
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (min(a,b), max(a,b))
                        if e in uncovered:
                            count += 1
                        else:
                            all_uncovered = False
                if all_uncovered and count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None or best_count == 0:
                e = next(iter(uncovered))
                best_clique = sorted([e[0], e[1]])
                best_count = 1
            
            cliques.append(best_clique)
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (min(a,b), max(a,b))
                    uncovered.discard(e)
        
        # Cover any remaining edges with 2-cliques
        for e in sorted(uncovered):
            cliques.append(sorted([e[0], e[1]]))
        
        return cliques
    
    def solve_simple(n, k, edges):
        edge_set = set((min(u,v), max(u,v)) for u, v in edges)
        C = set(range(k))
        cross_edges = [(u, v) for (u, v) in edge_set if (u in C and v not in C) or (v in C and u not in C)]
        cliques = [sorted(C)] + [sorted([u, v]) for u, v in cross_edges]
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        simple_cliques = solve_simple(n, k, edges)
        greedy_cliques = solve_greedy(n, k, edges)
        
        if len(greedy_cliques) <= len(simple_cliques):
            results.append(greedy_cliques)
        else:
            results.append(simple_cliques)
    
    final = []
    for clique_list in results:
        final.append([list(int(v) for v in clique) for clique in clique_list])
    
    return final
