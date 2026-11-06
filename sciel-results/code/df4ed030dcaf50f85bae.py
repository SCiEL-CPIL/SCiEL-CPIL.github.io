def construct():
    from random import Random
    import itertools
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj
    
    def partition_edges(edges, adj, n):
        if not edges:
            return []
        
        remaining = set(edges)
        result = []
        
        # Greedy: try to build larger cliques from uncovered edges
        # Start by trying to find triangles, then K4s, etc.
        
        # Build adjacency lookup
        edge_to_vertices = {}
        
        # Try to find maximal cliques greedily
        # For each uncovered edge, try to extend it to a larger clique
        
        while remaining:
            # Pick an edge, try to extend to largest clique
            # Sort edges to get deterministic behavior
            sorted_remaining = sorted(remaining)
            
            best_clique = None
            best_size = 0
            
            for e in sorted_remaining:
                u, v = e
                # Try to extend {u, v} to a clique
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                # Build clique starting with {u, v}
                clique = {u, v}
                # Greedily add vertices that connect to all in clique
                candidates = set(common)
                while candidates:
                    # Pick vertex with most connections to current clique
                    best_v = None
                    best_conn = -1
                    for v2 in sorted(candidates):
                        conn = len(clique & adj[v2])
                        if conn > best_conn:
                            best_conn = conn
                            best_v = v2
                    if best_v is not None and best_conn == len(clique):
                        clique.add(best_v)
                        candidates = candidates & adj[best_v]
                    else:
                        break
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique and len(best_clique) >= 2:
                # Add this clique
                result.append(sorted(best_clique))
                # Remove all edges within this clique
                for u, v in itertools.combinations(best_clique, 2):
                    if u < v:
                        remaining.discard((u, v))
                    else:
                        remaining.discard((v, u))
            else:
                break
        
        # Add remaining edges as size-2 cliques
        for e in sorted(remaining):
            result.append(list(e))
        
        return result
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges, adj = instance(n, p, seed)
        cliques = partition_edges(edges, adj, n)
        results.append(cliques)
    
    return results
