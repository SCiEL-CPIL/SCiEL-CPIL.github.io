import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    def merge_paths(paths):
        """Merge paths that share an endpoint, with multiple passes."""
        changed = True
        while changed:
            changed = False
            new_paths = []
            used = [False] * len(paths)
            
            for i in range(len(paths)):
                if used[i]:
                    continue
                p_i = paths[i]
                if len(p_i) < 2:
                    new_paths.append(p_i)
                    continue
                    
                merged_any = False
                for j in range(i + 1, len(paths)):
                    if used[j]:
                        continue
                    p_j = paths[j]
                    
                    merged = None
                    if p_i[-1] == p_j[0]:
                        candidate = p_i + p_j[1:]
                        if len(candidate) == len(set(candidate)):
                            merged = candidate
                    if merged is None and p_i[-1] == p_j[-1]:
                        candidate = p_i + list(reversed(p_j[1:]))
                        if len(candidate) == len(set(candidate)):
                            merged = candidate
                    if merged is None and p_i[0] == p_j[0]:
                        candidate = list(reversed(p_i[1:])) + p_j
                        if len(candidate) == len(set(candidate)):
                            merged = candidate
                    if merged is None and p_i[0] == p_j[-1]:
                        candidate = p_i[1:] + list(reversed(p_j))
                        if len(candidate) == len(set(candidate)):
                            merged = candidate

                    if merged:
                        new_paths.append(merged)
                        used[i] = True
                        used[j] = True
                        changed = True
                        merged_any = True
                        break
                    
                if not used[i] and not merged_any:
                    new_paths.append(p_i)
            
            paths = new_paths
        
        return paths

    def local_search_refine(paths, edges, adj, n, time_budget=2.0):
        """Try to improve path cover by re-routing individual paths."""
        start = time.time()
        best_paths = paths[:]
        best_count = len(paths)
        
        for _ in range(50):
            if time.time() - start > time_budget:
                break
            if best_count == 1:
                break
            
            # Pick a random path to try to re-route
            idx = random.randint(0, len(best_paths) - 1)
            path = best_paths[idx]
            if len(path) < 3:
                continue
            
            # Try removing a middle vertex and reconnecting
            # Find a vertex in the middle of the path
            mid = len(path) // 2
            if mid <= 0 or mid >= len(path) - 1:
                continue
            
            v = path[mid]
            prev = path[mid - 1]
            next_v = path[mid + 1]
            
            # Try to find an alternative route from prev to next_v avoiding v
            # Simple: try direct edge
            if (min(prev, next_v), max(prev, next_v)) in edges:
                # Check if this edge is available (not used by another path)
                # For simplicity, just try it
                new_path = path[:mid] + [next_v] + path[mid + 2:]
                # Validate
                valid = True
                for i in range(len(new_path) - 1):
                    u, w = new_path[i], new_path[i + 1]
                    e = (min(u, w), max(u, w))
                    if e not in edges:
                        valid = False
                        break
                if valid and len(new_path) == len(set(new_path)):
                    candidate = best_paths[:]
                    candidate[idx] = new_path
                    merged = merge_paths(candidate)
                    if len(merged) < best_count:
                        best_count = len(merged)
                        best_paths = merged
        
        return best_paths

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_vertices = [i for i in range(n) if deg[i] % 2 != 0]
        lower_bound = max(1, len(odd_vertices) // 2)
        
        best_paths = None
        best_count = float('inf')
        
        max_restarts = 80 if n <= 25 else 50

        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            if best_count == lower_bound:
                break
                
            curr_edges = set(edges)
            curr_paths = []
            rng_local = random.Random(r * 1000 + seed)
            
            # Endpoint-aware path extraction
            # Prioritize starting from odd-degree vertices
            start_priority = odd_vertices + [v for v in range(n) if v not in odd_vertices]
            rng_local.shuffle(start_priority[len(odd_vertices):])
            
            while curr_edges:
                if time.time() > deadline - 2.0:
                    break
                
                best_path = None
                best_len = 0
                
                # Compute current degrees
                deg_curr = defaultdict(int)
                for u, v in curr_edges:
                    deg_curr[u] += 1
                    deg_curr[v] += 1
                
                active_nodes = [u for u in deg_curr if deg_curr[u] > 0]
                if not active_nodes:
                    break
                
                # Sort: odd-degree first, then by degree
                active_nodes.sort(key=lambda x: (deg_curr[x] % 2 == 0, deg_curr[x]))
                
                # Try top starts
                k_starts = min(12, len(active_nodes))
                for s in active_nodes[:k_starts]:
                    # Beam search with endpoint awareness
                    beam = [([s], {s})]
                    beam_width = 15
                    
                    for d in range(n):
                        new_beam = []
                        for path, visited in beam:
                            u = path[-1]
                            candidates = []
                            for nb in adj[u]:
                                if nb in visited:
                                    continue
                                e = (min(u, nb), max(u, nb))
                                if e in curr_edges:
                                    # Score: prefer odd-degree vertices as endpoints
                                    # and vertices with fewer remaining connections
                                    rem_deg = sum(1 for m in adj[nb] 
                                                 if (min(nb, m), max(nb, m)) in curr_edges and m not in visited)
                                    is_odd = deg_curr[nb] % 2 != 0
                                    score = (0 if is_odd else 1, rem_deg)
                                    candidates.append((score, nb))
                            
                            if not candidates:
                                if len(path) > best_len:
                                    best_len = len(path)
                                    best_path = path[:]
                                continue
                            
                            candidates.sort()
                            for _, nb in candidates[:beam_width]:
                                new_beam.append((path + [nb], visited | {nb}))
                        
                        if not new_beam:
                            break
                            
                        new_beam.sort(key=lambda x: -len(x[0]))
                        beam = new_beam[:beam_width]
                        
                        if beam and len(beam[0][0]) > best_len:
                            best_len = len(beam[0][0])
                            best_path = beam[0][0][:]
                    
                if best_path and len(best_path) > 1:
                    valid = True
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i+1]
                        e = (min(u,v), max(u,v))
                        if e not in curr_edges:
                            valid = False
                            break
                    if valid:
                        curr_paths.append(best_path)
                        for i in range(len(best_path) - 1):
                            u, v = best_path[i], best_path[i+1]
                            e = (min(u,v), max(u,v))
                            curr_edges.discard(e)
                    else:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
                else:
                    if curr_edges:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
            
            if curr_paths:
                # Merge
                merged_paths = merge_paths(curr_paths)
                
                # Local search refinement
                if len(merged_paths) < best_count:
                    time_left_for_ls = min(2.0, deadline - 3.0 - time.time())
                    if time_left_for_ls > 0.5:
                        refined = local_search_refine(merged_paths, edges, adj, n, time_left_for_ls)
                        if len(refined) < len(merged_paths):
                            merged_paths = refined
                
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = merged_paths
                    
        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        all_results.append(best_paths)

    return all_results
