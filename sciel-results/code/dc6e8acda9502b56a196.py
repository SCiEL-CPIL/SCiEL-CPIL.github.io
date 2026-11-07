def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_maximal_clique(vertices, edges):
        """
        Find a maximal clique in the subgraph induced by 'vertices' using 'edges'.
        Greedy approach: Start with all vertices, remove those that don't form a clique.
        Since k is small, we can just try to build a large clique.
        """
        if not vertices:
            return []
        
        # Sort by degree in the induced subgraph to prioritize high-degree nodes?
        # Or just take the first one and expand?
        # For small k, simple greedy is fine.
        
        # Let's try to find a maximal clique by starting with the vertex with highest degree
        # in the induced subgraph, then adding neighbors that are connected to all current members.
        
        # Calculate degrees in induced subgraph
        deg = {v: 0 for v in vertices}
        for u in vertices:
            for w in vertices:
                if u < w and (u, w) in edges:
                    deg[u] += 1
                    deg[w] += 1
        
        # Sort vertices by degree descending
        sorted_verts = sorted(vertices, key=lambda x: -deg[x])
        
        clique = [sorted_verts[0]]
        for v in sorted_verts[1:]:
            # Check if v is connected to all in clique
            if all((min(v, u), max(v, u)) in edges for u in clique):
                clique.append(v)
        
        return clique

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        remaining_edges = set(edges)
        
        cliques = []
        
        # Process each vertex in I (independent set)
        for v in range(k, n):
            # Find neighbors of v in C that are still available (edge not covered)
            S_avail = []
            for u in range(k):
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    S_avail.append(u)
            
            if not S_avail:
                continue
            
            # Partition S_avail into cliques of the remaining graph
            # We want to minimize number of cliques, so we make them as large as possible
            while S_avail:
                # Find a maximal clique in the subgraph induced by S_avail
                clique_part = find_maximal_clique(S_avail, remaining_edges)
                if not clique_part:
                    # Should not happen if edges exist, but safety break
                    break
                
                # The full clique is {v} U clique_part
                full_clique = [v] + clique_part
                cliques.append(full_clique)
                
                # Remove all edges in this clique from remaining_edges
                for i in range(len(full_clique)):
                    for j in range(i + 1, len(full_clique)):
                        u, w = full_clique[i], full_clique[j]
                        e = (u, w) if u < w else (w, u)
                        remaining_edges.discard(e)
                
                # Remove clique_part from S_avail
                S_avail = [u for u in S_avail if u not in clique_part]
        
        # Process remaining edges (all in C)
        # Vertices are 0..k-1
        remaining_vertices = set(range(k))
        
        while remaining_edges:
            # Find a maximal clique in the remaining graph
            # We can just pick a vertex and expand
            # Or use the helper function on the set of vertices that have remaining edges
            
            # Get vertices involved in remaining edges
            verts_involved = set()
            for u, w in remaining_edges:
                verts_involved.add(u)
                verts_involved.add(w)
            
            if not verts_involved:
                break
            
            clique_part = find_maximal_clique(verts_involved, remaining_edges)
            if not clique_part:
                # Fallback: take any edge
                u, w = next(iter(remaining_edges))
                clique_part = [u, w]
            
            cliques.append(clique_part)
            
            # Remove edges
            for i in range(len(clique_part)):
                for j in range(i + 1, len(clique_part)):
                    u, w = clique_part[i], clique_part[j]
                    e = (u, w) if u < w else (w, u)
                    remaining_edges.discard(e)
            
            # If we didn't remove all edges, loop continues
            # But we must ensure progress.
            # If clique_part is size 1, we didn't remove any edge.
            # But find_maximal_clique returns at least 1 vertex.
            # If size 1, no edges removed.
            # We should ensure we pick at least 2 vertices if possible.
            # If remaining_edges is not empty, there is at least one edge (u, w).
            # So verts_involved has at least u, w.
            # find_maximal_clique should return at least {u, w} if (u, w) in edges.
            # So size >= 2.
            
        results.append(cliques)
    
    return results
