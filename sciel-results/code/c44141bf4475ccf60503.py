def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # 1. Identify C and I
        C = list(range(k))
        I = list(range(k, n))
        
        # 2. Build neighbor sets for I vertices in C
        neighbor_sets = {}
        for v in I:
            neighbors = [u for u in C if (u, v) in edge_set or (v, u) in edge_set]
            neighbor_sets[v] = set(neighbors)
        
        selected_cliques = []
        covered_cc_edges = set() # Edges in C x C covered by base cliques
        
        # 3. Create base cliques for each v in I
        for v in I:
            N_v = neighbor_sets[v]
            if len(N_v) >= 1:
                clique = sorted(list(N_v) + [v])
                selected_cliques.append(clique)
                # Mark C x C edges covered by this clique
                N_v_list = list(N_v)
                for i in range(len(N_v_list)):
                    for j in range(i + 1, len(N_v_list)):
                        u, w = N_v_list[i], N_v_list[j]
                        covered_cc_edges.add((min(u, w), max(u, w)))
        
        # 4. Identify remaining C x C edges
        remaining_edges = set()
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                u, w = C[i], C[j]
                e = (min(u, w), max(u, w))
                if e in edge_set and e not in covered_cc_edges:
                    remaining_edges.add(e)
        
        # 5. Cover remaining edges with cliques from C
        # Build adjacency for remaining graph on C
        # Vertices are C
        # Edges are remaining_edges
        rem_adj = {u: set() for u in C}
        for u, w in remaining_edges:
            rem_adj[u].add(w)
            rem_adj[w].add(u)
            
        while remaining_edges:
            # Find a maximal clique in the remaining graph
            # Greedy: pick a vertex with max degree, try to expand
            # Or simply: pick any vertex, find common neighbors, etc.
            # Since it's a subgraph of a clique, we can just find a clique in rem_adj
            
            # Simple heuristic: pick vertex with highest degree
            if not rem_adj:
                break
                
            # Find vertex with max degree
            max_deg_vertex = max(rem_adj.keys(), key=lambda x: len(rem_adj[x]))
            
            # Start clique with this vertex
            clique_vertices = {max_deg_vertex}
            
            # Try to add other vertices
            # A vertex x can be added if it is connected to all vertices in clique_vertices
            # and the edge (x, y) is in remaining_edges for all y in clique_vertices
            # Actually, we just need the induced subgraph on clique_vertices to be a clique in the REMAINING graph.
            # So for any new x, (x, y) must be in remaining_edges for all y in clique_vertices.
            
            changed = True
            while changed:
                changed = False
                # Candidates are neighbors of all vertices in clique_vertices in the remaining graph
                candidates = set.intersection(*[rem_adj[v] for v in clique_vertices]) if clique_vertices else set(rem_adj.keys())
                for x in candidates:
                    if x not in clique_vertices:
                        # Check if x can be added
                        # It is in candidates, so it is connected to all in clique_vertices in the REMAINING graph
                        # So we can add it
                        clique_vertices.add(x)
                        changed = True
                        # Update candidates? 
                        # Actually, adding x might remove other candidates if they are not connected to x?
                        # No, candidates are those connected to ALL current vertices.
                        # If we add x, the new candidates must be connected to x as well.
                        # So we should recompute candidates.
                        candidates = set.intersection(*[rem_adj[v] for v in clique_vertices]) if clique_vertices else set(rem_adj.keys())
                        break # Restart loop with new clique
            
            # Convert to sorted list
            clique = sorted(list(clique_vertices))
            if len(clique) < 2:
                # Should not happen if remaining_edges is not empty, but just in case
                # Pick an edge
                u, w = next(iter(remaining_edges))
                clique = [u, w]
            
            selected_cliques.append(clique)
            
            # Remove edges covered by this clique from remaining_edges and rem_adj
            clique_set = set(clique)
            edges_to_remove = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    if e in remaining_edges:
                        edges_to_remove.add(e)
            
            for e in edges_to_remove:
                remaining_edges.discard(e)
                u, w = e
                rem_adj[u].discard(w)
                rem_adj[w].discard(u)
            
            # Clean up rem_adj if any vertices have no edges
            # Not strictly necessary for correctness, but good for efficiency
            
        return [list(c) for c in selected_cliques]
    
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    return [clique_partition(*spec) for spec in specs]
