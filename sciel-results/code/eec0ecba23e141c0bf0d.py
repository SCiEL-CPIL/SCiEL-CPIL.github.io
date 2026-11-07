def construct():
    from random import Random
    import itertools
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((u, v))
        return cross

    def max_clique(edges, nodes):
        # edges: set of tuples (u, v)
        # nodes: list of active nodes
        # Returns a list of nodes forming a maximum clique
        if not edges:
            return []
        
        # Convert to adjacency set for faster lookup
        adj = {u: set() for u in nodes}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Bron-Kerbosch with pivot for max clique
        # Since N is small, simple backtracking is fine
        best = []
        
        def expand(candidates, clique):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not candidates:
                return
            
            # Pick a pivot to reduce branching (optional for small N)
            # Just iterate
            for u in list(candidates):
                new_clique = clique + [u]
                new_candidates = candidates.intersection(adj[u])
                expand(new_candidates, new_clique)
                candidates.remove(u)
                
        # Start with all nodes that have edges
        start_nodes = set(nodes)
        # Actually, we should start with nodes that are part of the edge set
        active_nodes = set()
        for u, v in edges:
            active_nodes.add(u)
            active_nodes.add(v)
            
        expand(active_nodes, [])
        return best

    result = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cross = instance(n, k, p, seed)
        cliques = []
        
        # 1. Process independent set vertices
        # S_v is set of neighbors in C
        S = [set() for _ in range(n)]
        for (u, v) in cross:
            S[v].add(u)
            
        # Covered internal edges
        covered_internal = set()
        
        for v in range(k, n):
            if S[v]:
                clique = list(S[v]) + [v]
                cliques.append(clique)
                # Mark internal edges in S[v] as covered
                for u1, u2 in itertools.combinations(S[v], 2):
                    covered_internal.add((min(u1, u2), max(u1, u2)))
        
        # 2. Process remaining internal edges in C
        # Total internal edges in C
        all_internal = set(itertools.combinations(range(k), 2))
        remaining = all_internal - covered_internal
        
        # Greedy partition of remaining edges
        while remaining:
            # Find max clique in the graph defined by 'remaining'
            nodes_in_rem = set()
            for u, v in remaining:
                nodes_in_rem.add(u)
                nodes_in_rem.add(v)
                
            clique_nodes = max_clique(remaining, list(nodes_in_rem))
            
            if not clique_nodes:
                # Should not happen if remaining is not empty
                break
                
            # Add this clique
            cliques.append(clique_nodes)
            
            # Remove edges covered by this clique
            for u1, u2 in itertools.combinations(clique_nodes, 2):
                edge = (min(u1, u2), max(u1, u2))
                if edge in remaining:
                    remaining.remove(edge)
                    
        result.append(cliques)
        
    return result
