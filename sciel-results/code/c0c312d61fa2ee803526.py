import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_paths = None
        best_count = float('inf')
        
        # Try multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            rng = random.Random(seed * 1000 + r)
            remaining_edges = set(edges)
            paths = []
            
            # Helper to extend a path
            def extend_path(start, end, current_path, remaining):
                # Try to extend from end
                # Heuristic: Choose neighbor with min degree in remaining graph
                # We need to check if edge is in remaining
                # And if vertex is not in current_path
                
                # To speed up, we can just pick randomly or use degree
                # Let's use Warnsdorff: pick neighbor with lowest remaining degree
                
                # We need a way to check if a vertex is in path efficiently
                # Path is a list. For small n, linear search is ok.
                
                # Actually, we can just try to extend from both ends
                # Let's try to extend from 'end' first
                extended = False
                # Get neighbors of 'end'
                # Filter: edge (end, nb) in remaining, nb not in path
                candidates = []
                for nb in adj[end]:
                    e = tuple(sorted((end, nb)))
                    if e in remaining and nb not in current_path:
                        # Calculate remaining degree of nb
                        # deg(nb) in remaining graph
                        # This is expensive. Approximate with static degree?
                        # Or just count available edges from nb
                        avail = 0
                        for nn in adj[nb]:
                            e2 = tuple(sorted((nb, nn)))
                            if e2 in remaining:
                                avail += 1
                        candidates.append((avail, nb))
                        
                if candidates:
                    candidates.sort()
                    # Pick the one with fewest options
                    _, nb = candidates[0]
                    e = tuple(sorted((end, nb)))
                    remaining.remove(e)
                    current_path.append(nb)
                    return True, current_path, remaining
                
                # Try to extend from 'start'
                candidates = []
                for nb in adj[start]:
                    e = tuple(sorted((start, nb)))
                    if e in remaining and nb not in current_path:
                        avail = 0
                        for nn in adj[nb]:
                            e2 = tuple(sorted((nb, nn)))
                            if e2 in remaining:
                                avail += 1
                        candidates.append((avail, nb))
                        
                if candidates:
                    candidates.sort()
                    _, nb = candidates[0]
                    e = tuple(sorted((start, nb)))
                    remaining.remove(e)
                    current_path.insert(0, nb)
                    return True, current_path, remaining
                
                return False, current_path, remaining

            while remaining_edges:
                if time.time() > deadline:
                    break
                    
                # Pick a random edge to start
                e = rng.choice(list(remaining_edges))
                u, v = e
                remaining_edges.remove(e)
                path = [u, v]
                
                # Extend
                while True:
                    extended, path, remaining_edges = extend_path(path[0], path[-1], path, remaining_edges)
                    if not extended:
                        break
                        
                paths.append(path)
                
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
        return best_paths

    results = []
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return single edge paths for remaining
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(*spec)
        results.append(paths)
        
    return results

# Need to import Random
from random import Random
