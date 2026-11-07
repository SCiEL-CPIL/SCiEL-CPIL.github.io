def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges
        # N[v] is set of neighbors in C for v in I
        N = [set() for _ in range(n)]
        # internal edges of C are all pairs (u, v) with u < v < k
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                internal_edges.add((u, v))
                
        cross_edges = []
        for u, v in edges:
            if u < k and v < k:
                continue # Already handled in internal_edges
            else:
                # u < k, v >= k
                cross_edges.append((u, v))
                N[v].add(u)
                N[u].add(v) # Not strictly needed for C vertices, but good for symmetry if needed, though we only care about N[v] for v in I

        cliques = []
        
        # 1. Create cliques for active I vertices
        active_I = []
        for v in range(k, n):
            if N[v]:
                active_I.append(v)
                # Clique is {v} U N[v]
                clique = [v] + sorted(N[v])
                cliques.append(clique)
                
        # 2. Determine covered internal edges
        # An internal edge (u, w) is covered if there exists v in active_I 
        # such that u in N[v] and w in N[v]
        covered_internal = set()
        for v in active_I:
            neighbors = list(N[v])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u, w = neighbors[i], neighbors[j]
                    if u > w: u, w = w, u
                    covered_internal.add((u, w))
                    
        # 3. Find uncovered internal edges
        uncovered_internal = internal_edges - covered_internal
        
        # 4. Cover uncovered_internal with cliques (subsets of C)
        # Graph G_U has vertices 0..k-1 and edges uncovered_internal
        # We need to partition these edges into cliques.
        
        if uncovered_internal:
            # Greedy clique cover
            # Convert to adjacency matrix or set for fast lookup
            remaining_edges = set(uncovered_internal)
            
            while remaining_edges:
                # Find a maximal clique in the graph defined by remaining_edges
                # Since k is small, we can pick an edge, try to extend it
                # Or simply: pick a vertex with highest degree in remaining graph, 
                # find its neighbors in remaining graph, find maximal clique there.
                
                # For simplicity and correctness on small graphs:
                # Pick an arbitrary edge (u, w)
                # Try to find the largest clique containing u and w in the remaining graph.
                # Or just greedy: 
                # 1. Pick vertex v with max degree in remaining graph.
                # 2. Let Cands = neighbors of v in remaining graph.
                # 3. Find maximal clique in Cands (in the induced subgraph).
                # 4. Clique = {v} U MaxClique.
                
                # Helper to find maximal clique in a set of vertices with given edge set
                def find_maximal_clique(vertices, edge_set):
                    if not vertices:
                        return []
                    # Greedy: pick vertex, recurse on neighbors
                    # This is not guaranteed largest, but finds a maximal one.
                    # Given small size, it's fine.
                    v = next(iter(vertices))
                    neighbors_in_subgraph = set()
                    for u, w in edge_set:
                        if u == v and w in vertices:
                            neighbors_in_subgraph.add(w)
                        elif w == v and u in vertices:
                            neighbors_in_subgraph.add(u)
                    
                    # Recursively find maximal clique in neighbors
                    sub_clique = find_maximal_clique(neighbors_in_subgraph, edge_set)
                    return [v] + sub_clique

                # Get vertices present in remaining edges
                verts_in_rem = set()
                for u, w in remaining_edges:
                    verts_in_rem.add(u)
                    verts_in_rem.add(w)
                    
                # Find a maximal clique
                clique = find_maximal_clique(verts_in_rem, remaining_edges)
                
                if not clique:
                    # Should not happen if remaining_edges is not empty
                    break
                    
                # Add clique to results
                cliques.append(sorted(clique))
                
                # Remove edges covered by this clique
                clique_set = set(clique)
                new_remaining = set()
                for u, w in remaining_edges:
                    if u in clique_set and w in clique_set:
                        continue # Covered
                    else:
                        new_remaining.add((u, w))
                remaining_edges = new_remaining

        results.append(cliques)
        
    return results
