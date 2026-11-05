"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from collections import defaultdict, deque
    import random
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are connected by an edge."""
        m = len(edges)
        conflict = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check shared vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if any endpoint of e1 is adjacent to any endpoint of e2
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict
    
    def max_induced_matching_bipartite(edges, adj, remaining_edges):
        """
        Find a maximum induced matching in the subgraph induced by remaining_edges.
        We model this as: for each edge in remaining_edges, we need to select a subset
        such that no two selected edges conflict. This is a maximum independent set in
        the conflict graph restricted to remaining_edges.
        
        Since exact maximum independent set is NP-hard, we use a greedy approach
        with a twist: we try to maximize the size by processing edges in an order
        that favors edges with fewer conflicts among remaining edges.
        
        Actually, let's use a different approach: iteratively pick the edge with
        minimum degree in the remaining conflict graph, add it to the matching,
        and remove all its neighbors. This is a greedy maximal independent set
        heuristic that tends to produce larger independent sets.
        """
        # Build conflict graph for remaining edges only
        rem_set = set(remaining_edges)
        rem_indices = [i for i in range(len(edges)) if i in rem_set]
        rem_to_local = {i: k for k, i in enumerate(rem_indices)}
        n_rem = len(rem_indices)
        
        if n_rem == 0:
            return []
        
        # Build local conflict graph
        local_conflict = [set() for _ in range(n_rem)]
        for ki, i in enumerate(rem_indices):
            for j in rem_indices:
                if j > i and j in local_conflict[ki] is False:  # placeholder
                    pass
            # Better: iterate over pairs
        for ki, i in enumerate(rem_indices):
            for kj, j in enumerate(rem_indices):
                if ki != kj and j in conflict_global[i]:
                    local_conflict[ki].add(kj)
        
        # Greedy: repeatedly pick vertex with min degree in remaining conflict graph
        selected = []
        active = set(range(n_rem))
        
        while active:
            # Find vertex with minimum degree in active subgraph
            min_deg = float('inf')
            chosen = None
            for v in active:
                deg = len(local_conflict[v] & active)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            
            if chosen is None:
                break
                
            selected.append(chosen)
            # Remove chosen and its neighbors from active
            active.discard(chosen)
            for nb in local_conflict[chosen]:
                active.discard(nb)
        
        # Convert back to global edge indices
        return [rem_indices[i] for i in selected]
    
    def greedy_coloring_improved(conflict, m):
        """
        Improved coloring: iteratively find a large induced matching (independent set
        in conflict graph), assign it a new color, remove those edges, repeat.
        """
        color = [-1] * m
        remaining = set(range(m))
        color_idx = 0
        
        while remaining:
            # Find a large independent set in the conflict graph restricted to remaining
            rem_list = list(remaining)
            
            # Build local conflict
            rem_set = set(rem_list)
            rem_to_local = {i: k for k, i in enumerate(rem_list)}
            n_rem = len(rem_list)
            local_conflict = [set() for _ in range(n_rem)]
            
            for ki, i in enumerate(rem_list):
                for j in conflict[i]:
                    if j in rem_set:
                        kj = rem_to_local[j]
                        local_conflict[ki].add(kj)
            
            # Greedy maximal independent set with min-degree ordering
            selected = []
            active = set(range(n_rem))
            
            while active:
                min_deg = float('inf')
                chosen = None
                for v in active:
                    deg = len(local_conflict[v] & active)
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                
                if chosen is None:
                    break
                    
                selected.append(chosen)
                active.discard(chosen)
                for nb in local_conflict[chosen]:
                    active.discard(nb)
            
            # Assign color to selected edges
            for ki in selected:
                global_idx = rem_list[ki]
                color[global_idx] = color_idx
                remaining.discard(global_idx)
            
            color_idx += 1
        
        # Renumber colors to be 0..k-1 (already done since we assign sequentially)
        return color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build conflict graph
        conflict = build_conflict_graph(edges, adj)
        
        # Store in global for use in helper (not ideal, but works for this scope)
        global conflict_global
        conflict_global = conflict
        
        # Greedy coloring with max independent set per color class
        color = greedy_coloring_improved(conflict, m)
        
        out.append(color)
    
    return out
# EVOLVE-BLOCK-END
