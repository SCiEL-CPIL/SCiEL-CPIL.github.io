def construct():
    import time
    start_time = time.time()
    DEADLINE = 115

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        neighbors_in_C = {v: set() for v in range(k, n)}
        for (u, v) in edges:
            if u < k and v < k:
                continue
            elif u >= k and v >= k:
                continue
            else:
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                neighbors_in_C[i].add(c)
        
        # Track which C-edges are already covered
        covered_c_edges = set()
        cliques = []
        
        # For each I-vertex, form a clique with it and some of its C-neighbors
        # To avoid conflicts, process I-vertices and greedily assign C-vertices
        # Each C-vertex can be "shared" across multiple I-cliques only if the C-edges it forms 
        # are not double-counted.
        
        # Simple approach: for each v in I, use clique {v} union N_C(v) if no C-edge within N_C(v) 
        # is already covered. Otherwise, fall back to edge-cliques for v.
        
        for v in range(k, n):
            nc = neighbors_in_C[v]
            if not nc:
                continue
            
            nc_list = sorted(nc)
            # Check if all C-edges within nc are uncovered
            can_merge = True
            for i in range(len(nc_list)):
                for j in range(i + 1, len(nc_list)):
                    e = (nc_list[i], nc_list[j])
                    if e in covered_c_edges:
                        can_merge = False
                        break
                if not can_merge:
                    break
            
            if can_merge and len(nc_list) >= 1:
                # Use merged clique
                clique = nc_list + [v]
                cliques.append(clique)
                # Mark C-edges within nc as covered
                for i in range(len(nc_list)):
                    for j in range(i + 1, len(nc_list)):
                        covered_c_edges.add((nc_list[i], nc_list[j]))
            else:
                # Use edge-cliques
                for c in nc_list:
                    cliques.append([c, v])
        
        # Cover remaining C-edges
        remaining_c_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_c_edges:
                    remaining_c_edges.add((u, v))
        
        # Greedy clique cover for remaining C-edges
        while remaining_c_edges:
            edge = next(iter(remaining_c_edges))
            clique_vertices = set(edge)
            remaining_c_edges.discard(edge)
            changed = True
            while changed:
                changed = False
                for v in range(k):
                    if v in clique_vertices:
                        continue
                    can_add = True
                    for u in clique_vertices:
                        e = (min(u, v), max(u, v))
                        if e not in remaining_c_edges:
                            can_add = False
                            break
                    if can_add:
                        clique_vertices.add(v)
                        for u in list(clique_vertices):
                            if u == v:
                                continue
                            e = (min(u, v), max(u, v))
                            if e in remaining_c_edges:
                                remaining_c_edges.discard(e)
                        changed = True
            if len(clique_vertices) >= 2:
                cliques.append(sorted(clique_vertices))
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() - start_time > DEADLINE:
            break
        results.append(solve_one(*spec))
    
    while len(results) < 3:
        spec = specs[len(results)]
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        for (u, v) in sorted(edges):
            if u < k and v < k:
                continue
            cliques.append([min(u, v), max(u, v)])
        results.append(cliques)
    
    return results
