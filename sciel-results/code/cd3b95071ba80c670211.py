# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict

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
        m = len(edges)
        # Use bitsets or sets for conflicts
        # Since m is small, sets are fine
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    # Check adjacency
                    if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                        u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                        conflict[i].add(j)
                        conflict[j].add(i)
        return conflict

    def find_max_induced_matching(conflict, remaining):
        """
        Find a large independent set in the conflict graph restricted to 'remaining'.
        Uses Greedy + Local Search (1-opt and 2-opt).
        """
        rem_list = list(remaining)
        n_rem = len(rem_list)
        if n_rem == 0:
            return set()
        
        # Map global to local
        # Actually, we can just work with global indices and use the 'remaining' set for validity
        # But local indices are faster for bit operations if we used them.
        # Let's stick to global indices for clarity, using sets for conflict checks.
        
        best_set = set()
        
        # Run multiple greedy starts
        for _ in range(10): # 10 restarts
            # Greedy Initialization
            current_set = set()
            active = set(remaining)
            
            while active:
                # Choose vertex with min degree in active subgraph
                min_deg = float('inf')
                chosen = None
                for v in active:
                    deg = len(conflict[v] & active)
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                    elif deg == min_deg:
                        # Random tie-breaking
                        if chosen is None or random.random() < 0.5:
                            chosen = v
                
                if chosen is None:
                    break
                
                current_set.add(chosen)
                active.discard(chosen)
                for nb in conflict[chosen]:
                    active.discard(nb)
            
            # Local Search Improvement
            improved = True
            while improved:
                improved = False
                
                # 1-opt: Add a vertex
                for v in list(remaining):
                    if v not in current_set:
                        # Check if v conflicts with any in current_set
                        if not (conflict[v] & current_set):
                            current_set.add(v)
                            improved = True
                
                # 2-opt: Remove 1, Add k > 1
                # To be efficient, we only try to remove vertices that are "expensive" (high degree in current context?)
                # Or just iterate all.
                for s in list(current_set):
                    temp_set = current_set - {s}
                    # Find candidates: vertices not in temp_set and not conflicting with temp_set
                    candidates = set()
                    for v in remaining:
                        if v in temp_set:
                            continue
                        if not (conflict[v] & temp_set):
                            candidates.add(v)
                    
                    # Find Max Independent Set in candidates (using greedy)
                    cand_set = set()
                    cand_active = set(candidates)
                    while cand_active:
                        min_deg = float('inf')
                        chosen = None
                        for v in cand_active:
                            deg = len(conflict[v] & cand_active)
                            if deg < min_deg:
                                min_deg = deg
                                chosen = v
                        if chosen is None:
                            break
                        cand_set.add(chosen)
                        cand_active.discard(chosen)
                        for nb in conflict[chosen]:
                            cand_active.discard(nb)
                    
                    if len(cand_set) > 1:
                        current_set = temp_set | cand_set
                        improved = True
                        break # Restart the loop as set changed significantly
            
            if len(current_set) > len(best_set):
                best_set = current_set
        
        return best_set

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_graph(edges, adj)
        
        color = [-1] * m
        remaining = set(range(m))
        color_idx = 0
        
        while remaining:
            selected = find_max_induced_matching(conflict, remaining)
            for idx in selected:
                color[idx] = color_idx
                remaining.discard(idx)
            color_idx += 1
        
        out.append(color)
    
    return out
# EVOLVE-BLOCK-END
