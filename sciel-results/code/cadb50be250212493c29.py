import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit

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

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Calculate lower bound
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 != 0)
        lower_bound = max(1, odd_count // 2)
        
        best_paths = None
        best_count = float('inf')
        
        # Restart counts
        if n <= 20:
            max_restarts = 100
        elif n <= 25:
            max_restarts = 50
        else:
            max_restarts = 30

        def can_merge(p1, p2):
            """Check if p1 and p2 can be merged into a simple path. Returns merged path or None."""
            # Case 1: p1 end == p2 start
            if p1[-1] == p2[0]:
                candidate = p1 + p2[1:]
                if len(candidate) == len(set(candidate)):
                    return candidate
            # Case 2: p1 end == p2 end
            if p1[-1] == p2[-1]:
                candidate = p1 + list(reversed(p2[1:]))
                if len(candidate) == len(set(candidate)):
                    return candidate
            # Case 3: p1 start == p2 start
            if p1[0] == p2[0]:
                candidate = list(reversed(p1[1:])) + p2
                if len(candidate) == len(set(candidate)):
                    return candidate
            # Case 4: p1 start == p2 end
            if p1[0] == p2[-1]:
                candidate = p1[1:] + list(reversed(p2))
                if len(candidate) == len(set(candidate)):
                    return candidate
            return None

        def optimize_paths(paths, adj, time_limit):
            """Local search: Try to merge paths. If stuck, try splitting and re-merging."""
            current_paths = [list(p) for p in paths]
            best_local = [list(p) for p in paths]
            best_local_count = len(current_paths)
            
            # Pre-compute edge sets for fast lookup? 
            # For splitting, we need to know if a path is valid.
            
            def is_valid_path(path):
                if len(path) < 2: return True
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if (min(u,v), max(u,v)) not in edge_set_global:
                        return False
                return len(path) == len(set(path))

            # We need a global edge set for this graph instance
            # Pass it in or capture it. Let's pass it.
            
            while time.time() < time_limit:
                # 1. Try to merge any pair
                merged = False
                for i in range(len(current_paths)):
                    if len(current_paths) <= best_local_count:
                        break # No point merging if we already improved and can't go lower? 
                        # Actually, we want to reach lower_bound. 
                        # If current_count == best_local_count, we might still find a better arrangement?
                        # No, merging reduces count. We just want to merge as much as possible.
                    for j in range(i + 1, len(current_paths)):
                        p1 = current_paths[i]
                        p2 = current_paths[j]
                        merged_path = can_merge(p1, p2)
                        if merged_path:
                            # Replace i and j with merged_path
                            current_paths = [current_paths[k] for k in range(len(current_paths)) if k != i and k != j]
                            current_paths.append(merged_path)
                            merged = True
                            if len(current_paths) < best_local_count:
                                best_local_count = len(current_paths)
                                best_local = [list(p) for p in current_paths]
                            break
                    if merged:
                        break
                
                if not merged:
                    # 2. If no merges possible, try a random split
                    # Only split if we have time and haven't improved recently?
                    # For now, always try split if stuck, to escape local minima.
                    if len(current_paths) > lower_bound:
                        # Pick a random path with length > 1
                        splittable = [i for i, p in enumerate(current_paths) if len(p) > 1]
                        if splittable:
                            idx = random.choice(splittable)
                            path = current_paths[idx]
                            # Pick a random edge to split at
                            # path is [v0, v1, ..., vk]
                            # Split between path[i] and path[i+1]
                            split_idx = random.randint(0, len(path) - 2)
                            p1 = path[:split_idx + 1]
                            p2 = path[split_idx + 1:]
                            
                            # Remove original, add split parts
                            current_paths.pop(idx)
                            current_paths.append(p1)
                            current_paths.append(p2)
                            
                            # Immediately try to merge the new parts with others
                            # (The loop will restart and try to merge)
                            # To avoid infinite loops of split-merge-split, we rely on the 
                            # fact that we keep the best_local.
                            # But we need to ensure we don't loop forever.
                            # The "merged" flag was False, so we did a split.
                            # Next iteration will try to merge.
                            # If the split parts merge back, we are back to start.
                            # To avoid this, we can bias the split or randomize more.
                            # For now, this simple heuristic might oscillate.
                            # Add a counter or randomization?
                            # Let's just let it run. The time limit stops it.
                            
                    else:
                        break # Can't improve below lower bound, and no merges? 
                        # Actually, if len == lower_bound, we are optimal.
            
            return best_local

        # We need edge_set_global for the optimizer
        edge_set_global = set(edges)

        for r in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
            if best_count == lower_bound:
                break
                
            curr_edges = set(edges)
            curr_paths = []
            
            # Randomize tie-breaking
            rng_local = random.Random(r)
            
            # Greedy Extraction
            while curr_edges:
                if time.time() > deadline - 5.0:
                    break
                
                best_path = None
                best_len = 0
                
                # Calculate degrees
                deg_curr = defaultdict(int)
                for u, v in curr_edges:
                    deg_curr[u] += 1
                    deg_curr[v] += 1
                
                active_nodes = [u for u in deg_curr if deg_curr[u] > 0]
                if not active_nodes:
                    break
                
                # Sort by degree
                active_nodes.sort(key=lambda x: deg_curr[x])
                
                # Try top k starts
                k_starts = 5
                for s in active_nodes[:k_starts]:
                    # Beam Search
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
                                    # Score: prefer neighbors with fewer remaining options
                                    score = 0
                                    for m in adj[nb]:
                                        if m in visited:
                                            continue
                                        e2 = (min(nb, m), max(nb, m))
                                        if e2 in curr_edges:
                                            score += 1
                                    # Add noise
                                    score += rng_local.random() * 0.5
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
                    # Verify
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
                # Optimize Phase
                # Allocate some time for optimization
                opt_time_limit = time.time() + 2.0 
                if time.time() > deadline - 10.0:
                    opt_time_limit = deadline - 10.0
                    
                optimized_paths = optimize_paths(curr_paths, adj, opt_time_limit)
                
                if len(optimized_paths) < best_count:
                    best_count = len(optimized_paths)
                    best_paths = optimized_paths

        if best_paths is None:
            best_paths = [list(e) for e in edges]
            
        all_results.append(best_paths)

    return all_results
