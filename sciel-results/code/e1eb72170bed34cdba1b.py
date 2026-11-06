import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # 20s margin
    
    # Generate the three graphs
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    graphs = [instance(*spec) for spec in specs]
    
    results = []
    
    for edges in graphs:
        if time.time() > deadline:
            # Fallback: return trivial solution for remaining
            results.append([list(e) for e in edges])
            continue
            
        # Initial paths: each edge is a path
        paths = [list(e) for e in edges]
        
        # Helper to check if a list of vertices is a simple path in the graph
        # We assume the edges used are valid and we just need to check vertex uniqueness
        def is_simple_path(path):
            return len(set(path)) == len(path)
        
        # Helper to try merging two paths
        # Returns the merged path if valid, else None
        def try_merge(p1, p2):
            # Try p1 + p2 (last of p1 == first of p2)
            if p1[-1] == p2[0]:
                merged = p1 + p2[1:]
                if is_simple_path(merged):
                    return merged
            # Try p2 + p1
            if p2[-1] == p1[0]:
                merged = p2 + p1[1:]
                if is_simple_path(merged):
                    return merged
            # Try p1 + reversed(p2) (last of p1 == last of p2)
            if p1[-1] == p2[-1]:
                merged = p1 + p2[-2::-1]
                if is_simple_path(merged):
                    return merged
            # Try p2 + reversed(p1)
            if p2[-1] == p1[-1]:
                merged = p2 + p1[-2::-1]
                if is_simple_path(merged):
                    return merged
            return None
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to find any valid merge
            n_paths = len(paths)
            found_merge = False
            # To speed up, we can index paths by their endpoints
            # But n is small, O(n^2) is fine
            for i in range(n_paths):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_paths):
                    merged = try_merge(paths[i], paths[j])
                    if merged is not None:
                        # Perform merge
                        paths.pop(j)
                        paths[i] = merged
                        improved = True
                        found_merge = True
                        break
                if found_merge:
                    break
            
            if not improved and time.time() < deadline:
                # Perturbation: Split a random path to allow different merges
                if len(paths) > 1:
                    idx = random.randint(0, len(paths) - 1)
                    p = paths[idx]
                    if len(p) > 2:
                        # Split at a random internal vertex
                        split_idx = random.randint(1, len(p) - 2)
                        p1 = p[:split_idx + 1]
                        p2 = p[split_idx:]
                        paths.pop(idx)
                        paths.append(p1)
                        paths.append(p2)
                        improved = True # Force another iteration
        
        results.append(paths)
        
    return results
