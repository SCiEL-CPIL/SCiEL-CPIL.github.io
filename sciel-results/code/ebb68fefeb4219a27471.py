def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        C = set(range(k))
        I = set(range(k, n))
        
        cliques = []
        remaining = set(edge_set)
        
        # Precompute neighbors in C for each v in I
        # Actually, we can just iterate through I and find neighbors in remaining
        
        # Process vertices in I
        # Sort I by degree in remaining? Or just iterate?
        # Iterating and taking maximal clique {v} U N(v) is a good heuristic.
        # To be safe, let's process I vertices that have the most edges in remaining first?
        # Or just process all.
        
        # Let's compute the neighborhood in C for each v in I based on the ORIGINAL graph?
        # No, we need to remove edges as we go.
        # But note: Edges between I vertices do not exist.
        # So edges incident to v in I are only (v, u) for u in C.
        # When we form clique {v} U S, we remove (v, u) for u in S and (u, w) for u,w in S.
        # The edges (v, u) for u NOT in S remain.
        # So we must ensure S is a clique in the remaining graph?
        # S is a subset of C. In the original graph, C is a clique.
        # In the remaining graph, some edges within C might be removed by previous cliques.
        # So S must be a clique in the REMAINING graph.
        
        # Strategy:
        # 1. For each v in I:
        #    - Find N_rem(v) = {u in C | (min(v,u), max(v,u)) in remaining}
        #    - Find a maximal clique S in the subgraph of remaining induced by N_rem(v).
        #    - Clique is {v} U S.
        #    - Remove edges of this clique from remaining.
        
        # To find maximal clique in N_rem(v), we can use a simple greedy:
        # Pick u in N_rem(v) with max degree in remaining induced by N_rem(v).
        # Add to S. Remove its non-neighbors from candidates.
        
        def find_maximal_clique_in_set(vertices, rem_edges):
            if not vertices:
                return []
            # vertices is a set of ints
            # rem_edges is a set of tuples
            # We want a maximal clique in the graph induced by vertices with edges from rem_edges
            
            # Sort by degree in induced subgraph
            deg = {v: 0 for v in vertices}
            for u, w in rem_edges:
                if u in vertices and w in vertices:
                    deg[u] += 1
                    deg[w] += 1
            
            sorted_v = sorted(vertices, key=lambda x: deg[x], reverse=True)
            clique = []
            for v in sorted_v:
                # Check if v connects to all in clique
                ok = True
                for c in clique:
                    e = (min(v, c), max(v, c))
                    if e not in rem_edges:
                        ok = False
                        break
                if ok:
                    clique.append(v)
            return clique

        for v in sorted(I): # Sort for determinism, or maybe by degree?
            # Find neighbors in C in remaining
            nbrs = set()
            for u, w in remaining:
                if u == v:
                    nbrs.add(w)
                elif w == v:
                    nbrs.add(u)
            
            if not nbrs:
                continue
                
            # Find maximal clique S in nbrs using remaining edges
            S = find_maximal_clique_in_set(nbrs, remaining)
            
            if S:
                clique = [v] + list(S)
                cliques.append(sorted(clique))
                
                # Remove edges
                to_remove = set()
                # Edges (v, u) for u in S
                for u in S:
                    to_remove.add((min(v, u), max(v, u)))
                # Edges within S
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        u, w = S[i], S[j]
                        to_remove.add((min(u, w), max(u, w)))
                        
                remaining -= to_remove

        # Process remaining edges (all in C)
        while remaining:
            # Find a vertex with max degree in remaining
            deg = {}
            for u, w in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[w] = deg.get(w, 0) + 1
            
            start = max(deg, key=lambda x: deg[x])
            
            # Find maximal clique containing start in remaining
            # Candidates are neighbors of start in remaining
            cand = set()
            for u, w in remaining:
                if u == start:
                    cand.add(w)
                elif w == start:
                    cand.add(u)
            
            # Greedy build clique
            clique = [start]
            # Sort candidates by degree in remaining
            cands_list = list(cand)
            cands_list.sort(key=lambda x: deg.get(x, 0), reverse=True)
            
            for c in cands_list:
                connected = True
                for m in clique:
                    if (min(c, m), max(c, m)) not in remaining:
                        connected = False
                        break
                if connected:
                    clique.append(c)
            
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                to_remove = set()
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, v = clique[i], clique[j]
                        to_remove.add((min(u, v), max(u, v)))
                remaining -= to_remove
            else:
                # Fallback: single edge
                u, v = next(iter(remaining))
                cliques.append([u, v])
                remaining.discard((min(u, v), max(u, v)))
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        # Ensure types are correct
        # cliques is a list of lists of ints
        results.append(cliques)
        
    return results
