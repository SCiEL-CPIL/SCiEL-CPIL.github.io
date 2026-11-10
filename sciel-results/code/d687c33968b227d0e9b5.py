"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Build adjacency for K and I
        # K is 0..k-1, I is k..n-1
        # adj_K[u] = set of neighbors in K
        # adj_I[v] = set of neighbors in K (v is in I)
        
        adj_K = [set() for _ in range(k)]
        adj_I = [set() for _ in range(n - k)] # indexed by v-k
        
        for u, v in edges:
            if u < k and v < k:
                adj_K[u].add(v)
                adj_K[v].add(u)
            elif u < k and v >= k:
                adj_I[v - k].add(u)
            elif u >= k and v < k:
                adj_I[u - k].add(v)
            
        cliques = []
        covered_inner = set() # set of (u, w) with u < w, u, w in K
        
        # Process each vertex in I
        for i, v in enumerate(range(k, n)):
            N_v = adj_I[i]
            if not N_v:
                continue
                
            # We need to partition N_v into sets S_j such that
            # each S_j is an independent set in the graph of covered_inner edges restricted to N_v.
            
            # Build graph H_F on N_v
            # Vertices are in N_v
            # Edge (u, w) in H_F if (min(u,w), max(u,w)) in covered_inner
            
            # Greedy coloring of H_F
            # Colors are the sets S_j
            colors = {} # vertex -> color_id
            color_sets = [] # list of sets
            
            # Order vertices by degree in H_F (descending) for better coloring?
            # Or just arbitrary.
            verts = list(N_v)
            # Calculate degree in H_F for sorting
            degs = {}
            for u in verts:
                degs[u] = 0
            for u in verts:
                for w in verts:
                    if u < w and (u, w) in covered_inner:
                        degs[u] += 1
                        degs[w] += 1
            verts.sort(key=lambda x: degs[x], reverse=True)
            
            for u in verts:
                # Find smallest color that doesn't conflict
                used_colors = set()
                for w in verts:
                    if w == u: continue
                    if w in colors and (min(u, w), max(u, w)) in covered_inner:
                        used_colors.add(colors[w])
                        
                # Assign smallest available color
                c = 0
                while c in used_colors:
                    c += 1
                    
                colors[u] = c
                if c >= len(color_sets):
                    color_sets.append(set())
                color_sets[c].add(u)
                
            # Now we have color_sets. Each is an independent set in H_F.
            # Create cliques {v} U S_j
            for S in color_sets:
                if not S: continue
                Q = [v] + list(S)
                cliques.append(Q)
                # Update covered_inner
                S_list = list(S)
                for idx, u in enumerate(S_list):
                    for w in S_list[idx+1:]:
                        covered_inner.add((min(u, w), max(u, w)))
                        
        # Now cover remaining inner edges in K
        # Build graph G_R with vertices K and edges not in covered_inner
        # All possible inner edges are (u, w) with 0<=u<w<k
        # G_R edges are those NOT in covered_inner
        
        # We need to cover edges of G_R with cliques of G_R.
        # Greedy maximal clique cover
        
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_inner:
                    remaining_edges.add((u, w))
                    
        while remaining_edges:
            # Find a maximal clique in G_R
            # Start with a vertex of highest degree in G_R
            degs = {u: 0 for u in range(k)}
            for u, w in remaining_edges:
                degs[u] += 1
                degs[w] += 1
                
            # Pick vertex with max degree
            if not remaining_edges:
                break
            start_v = max(range(k), key=lambda x: degs[x])
            
            Q = {start_v}
            # Candidates are neighbors of start_v in G_R
            candidates = set()
            for u, w in remaining_edges:
                if u == start_v:
                    candidates.add(w)
                elif w == start_v:
                    candidates.add(u)
            
            # Greedily add vertices to Q
            while candidates:
                # Pick vertex in candidates with max degree in current induced subgraph?
                # Or just pick one
                # Let's pick one with max degree in remaining_edges restricted to Q U {c}
                # Simpler: just pick any, or max degree in candidates
                # For speed, just pick the first one or max degree
                # Let's try max degree in the subgraph induced by candidates
                c_degs = {c: 0 for c in candidates}
                for u, w in remaining_edges:
                    if u in candidates and w in candidates:
                        c_degs[u] += 1
                        c_degs[w] += 1
                        
                next_v = max(candidates, key=lambda x: c_degs[x])
                Q.add(next_v)
                # Remove next_v from candidates
                candidates.remove(next_v)
                # Remove non-neighbors of next_v from candidates
                # A vertex c is a neighbor of next_v in G_R if (min(c, next_v), max(c, next_v)) in remaining_edges
                new_candidates = set()
                for c in candidates:
                    if (min(c, next_v), max(c, next_v)) in remaining_edges:
                        new_candidates.add(c)
                candidates = new_candidates
                
            # Q is a clique in G_R
            # Add Q to cliques
            if len(Q) >= 2:
                cliques.append(list(Q))
            else:
                # If Q has 1 vertex, it covers no edges. But we must cover edges.
                # This case shouldn't happen if remaining_edges is not empty and we picked start_v from an edge.
                # But if start_v is isolated in G_R? No, we picked max degree, and if edges exist, degree > 0.
                pass
            
            # Remove edges covered by Q from remaining_edges
            Q_list = list(Q)
            to_remove = set()
            for i, u in enumerate(Q_list):
                for w in Q_list[i+1:]:
                    e = (min(u, w), max(u, w))
                    if e in remaining_edges:
                        to_remove.add(e)
            remaining_edges -= to_remove
            # Update covered_inner? Not strictly necessary for the loop, but good for consistency
            covered_inner |= to_remove

        results.append(cliques)
        
        if time.time() - start_time > 100:
            # Safety break, though unlikely
            break
            
    return results
# EVOLVE-BLOCK-END
