def construct():
    from random import Random
    import heapq

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = {i: set() for i in range(max(edges[0][0], edges[0][1]) + 1) if edges}
        # Re-initialize adj properly
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # Helper to check if edge is in remaining
        def has_edge(u, v):
            return tuple(sorted((u, v))) in remaining_edges

        while remaining_edges:
            # Find the longest simple path using a heuristic search
            best_path = []
            best_len = 0
            
            # Try starting from each vertex with remaining degree > 0
            # To speed up, limit trials or use a good heuristic
            candidates = [v for v in adj if any(has_edge(v, n) for n in adj.get(v, []))]
            
            # Sort candidates by degree descending to find good starts
            candidates.sort(key=lambda v: -len([n for n in adj.get(v, []) if has_edge(v, n)]))
            
            # Try a subset of starts if graph is large, otherwise all
            # For n=30, trying all is fine
            for start in candidates:
                if not any(has_edge(start, n) for n in adj.get(start, [])):
                    continue
                
                # DFS with heuristic (Warnsdorff-like)
                # We want to find a long path.
                # We'll do a greedy walk that prioritizes neighbors with fewer remaining options
                # to avoid trapping the path.
                
                def greedy_walk(start_vertex):
                    path = [start_vertex]
                    used_vertices = {start_vertex}
                    current = start_vertex
                    extended = True
                    while extended:
                        extended = False
                        neighbors = [n for n in adj.get(current, []) if n not in used_vertices and has_edge(current, n)]
                        if neighbors:
                            # Heuristic: pick neighbor with fewest remaining options (degree in remaining graph)
                            # This is Warnsdorff's rule
                            def get_degree(v):
                                return len([x for x in adj.get(v, []) if x not in used_vertices and has_edge(v, x)])
                            
                            # Sort by degree of the neighbor
                            neighbors.sort(key=get_degree)
                            next_v = neighbors[0]
                            path.append(next_v)
                            used_vertices.add(next_v)
                            remaining_edges.remove(tuple(sorted((current, next_v))))
                            current = next_v
                            extended = True
                    return path

                # Note: The above greedy_walk modifies remaining_edges globally if we don't copy it.
                # We need to simulate or rollback.
                # For simplicity and speed, we will perform the walk on a *copy* of the state?
                # That's expensive.
                # Alternative: Just do one greedy walk per start, pick the best, then commit.
                # But the walk modifies `remaining_edges`.
                
                # Let's restructure: 
                # 1. For each candidate start, simulate the walk on a local set of remaining edges?
                #    Too slow if we do it for all starts.
                # 2. Just pick the best start (highest degree) and do one walk?
                #    That's what the current code does essentially.
                
                # Let's try a slightly better heuristic: 
                # Try 5 random starts + best start.
                # And use a local set for simulation.
                
                pass # Placeholder for logic below

            # Implementation of the "Try multiple starts with local simulation"
            best_path_found = []
            best_path_len = 0
            
            # Select a few promising starts
            start_candidates = candidates[:5] if len(candidates) > 5 else candidates
            
            for start in start_candidates:
                # Local state for simulation
                sim_remaining = set(remaining_edges)
                sim_adj = {v: set() for v in adj}
                for u, v in sim_remaining:
                    sim_adj[u].add(v)
                    sim_adj[v].add(u)
                
                def sim_walk(s):
                    path = [s]
                    used = {s}
                    curr = s
                    while True:
                        neighbors = [n for n in sim_adj.get(curr, []) if n not in used and (tuple(sorted((curr, n))) in sim_remaining)]
                        if not neighbors:
                            break
                        # Heuristic: min degree in sim graph
                        def sim_deg(v):
                            return len([x for x in sim_adj.get(v, []) if x not in used and (tuple(sorted((v, x))) in sim_remaining)])
                        
                        neighbors.sort(key=sim_deg)
                        nxt = neighbors[0]
                        path.append(nxt)
                        used.add(nxt)
                        sim_remaining.remove(tuple(sorted((curr, nxt))))
                        # Update sim_adj? Not strictly necessary if we check sim_remaining
                        curr = nxt
                    return path

                p = sim_walk(start)
                if len(p) > best_path_len:
                    best_path_len = len(p)
                    best_path_found = p
                    # Save the remaining edges from this simulation?
                    # We need to know which edges were used.
                    # We can reconstruct them from the path.
            
            if best_path_found:
                paths.append(best_path_found)
                # Remove edges used in best_path_found from global remaining_edges
                for i in range(len(best_path_found) - 1):
                    u, v = best_path_found[i], best_path_found[i+1]
                    remaining_edges.remove(tuple(sorted((u, v))))
            else:
                # Fallback: pick any edge
                u, v = next(iter(remaining_edges))
                paths.append([u, v])
                remaining_edges.remove((u, v))

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
