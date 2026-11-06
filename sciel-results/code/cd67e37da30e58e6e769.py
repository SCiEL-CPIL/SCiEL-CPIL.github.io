import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    # Allocate time budget proportionally to expected difficulty (n^2)
    weights = [s[0]**2 for s in specs]
    total_weight = sum(weights)
    time_budget = deadline - start_time - 5.0  # 5s safety margin
    budgets = [time_budget * w / total_weight for w in weights]

    for spec_idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Build adjacency as lists for faster iteration
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute edge index for O(1) lookup using bitset
        # Edge (u,v) with u<v maps to index u*n + v
        def edge_key(u, v):
            if u > v:
                u, v = v, u
            return u * n + v
        
        # Use a set of edge keys for fast membership testing
        edge_key_set = set(edge_key(u, v) for u, v in edges)
        
        # Calculate lower bound
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 != 0)
        lower_bound = max(1, odd_count // 2)
        
        best_paths = None
        best_count = float('inf')
        
        # Heuristic: Greedy Longest Path with Randomized Restart
        max_restarts = 200
        if n <= 20:
            max_restarts = 400
        elif n <= 25:
            max_restarts = 300
        else:
            max_restarts = 200

        spec_deadline = start_time + budgets[spec_idx]
        if spec_deadline > deadline - 2.0:
            spec_deadline = deadline - 2.0

        for r in range(max_restarts):
            if time.time() > spec_deadline:
                break
            if best_count == lower_bound:
                break
                
            curr_edge_keys = set(edge_key_set)
            curr_paths = []
            
            # Randomize seed for tie-breaking in this restart
            rng_restart = random.Random(r + seed)
            
            while curr_edge_keys:
                if time.time() > spec_deadline - 1.0:
                    curr_paths = None
                    break
                
                # Find the longest simple path in curr_edges
                # We try starting from every vertex with degree > 0
                active_vertices = []
                for v in range(n):
                    has_edge = False
                    for nb in adj[v]:
                        if edge_key(v, nb) in curr_edge_keys:
                            has_edge = True
                            break
                    if has_edge:
                        active_vertices.append(v)
                
                if not active_vertices:
                    break
                
                best_path_in_iter = []
                
                # Try all active vertices as starting points
                for s in active_vertices:
                    # Greedy Warnsdorff from s
                    path = [s]
                    visited = {s}
                    curr_node = s
                    
                    while True:
                        candidates = []
                        for nb in adj[curr_node]:
                            if nb in visited:
                                continue
                            ek = edge_key(curr_node, nb)
                            if ek in curr_edge_keys:
                                # Calculate unvisited degree of nb in curr_edges
                                unvis_deg = 0
                                for m in adj[nb]:
                                    if m in visited:
                                        continue
                                    if edge_key(nb, m) in curr_edge_keys:
                                        unvis_deg += 1
                                candidates.append((unvis_deg, nb))
                        
                        if not candidates:
                            break
                        
                        # Sort by unvisited degree ascending (Warnsdorff)
                        candidates.sort(key=lambda x: x[0])
                        
                        # Randomize tie-breaking among those with minimum degree
                        min_deg = candidates[0][0]
                        min_candidates = [c for c in candidates if c[0] == min_deg]
                        next_node = rng_restart.choice(min_candidates)[1]
                        
                        path.append(next_node)
                        visited.add(next_node)
                        curr_node = next_node
                    
                    if len(path) > len(best_path_in_iter):
                        best_path_in_iter = path[:]

                if best_path_in_iter and len(best_path_in_iter) > 1:
                    curr_paths.append(best_path_in_iter)
                    # Remove edges from curr_edge_keys
                    for i in range(len(best_path_in_iter) - 1):
                        u, v = best_path_in_iter[i], best_path_in_iter[i + 1]
                        curr_edge_keys.discard(edge_key(u, v))
                else:
                    # Should not happen if curr_edge_keys is not empty, but safe fallback
                    if curr_edge_keys:
                        ek = next(iter(curr_edge_keys))
                        u, v = ek // n, ek % n
                        curr_paths.append([u, v])
                        curr_edge_keys.discard(ek)
            
            if curr_paths and len(curr_paths) < best_count:
                best_count = len(curr_paths)
                best_paths = curr_paths
                
        # If no best_paths found due to timeout, fallback to edge list
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        all_results.append(best_paths)

    return all_results
