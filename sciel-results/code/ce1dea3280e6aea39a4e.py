"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve_graph(n, p, seed, time_budget=30.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    # Build adjacency and degree structures
    adj = defaultdict(set)
    deg = [0] * n
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    # Fast Greedy Path Decomposition
    # Strategy: 
    # 1. Maintain a set of remaining edges.
    # 2. Repeatedly find a path by starting from a vertex with high remaining degree,
    #    and greedily extending it to a neighbor with high remaining degree,
    #    avoiding cycles (simple path constraint).
    # 3. Remove used edges.
    
    remaining_edges = set()
    for e in edges:
        remaining_edges.add((min(e[0], e[1]), max(e[0], e[1])))
        
    # Copy adjacency for quick removal
    rem_adj = defaultdict(set)
    for (u, v) in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    paths = []
    start_time = time.time()
    
    while remaining_edges:
        if time.time() - start_time > time_budget:
            # Fallback: single edges
            for e in remaining_edges:
                paths.append([e[0], e[1]])
            break
            
        # Find a starting vertex. 
        # Heuristic: Pick vertex with highest remaining degree.
        # If all degrees are 0, we are done (loop condition handles this).
        start_v = -1
        max_deg = -1
        for v in range(n):
            if len(rem_adj[v]) > max_deg:
                max_deg = len(rem_adj[v])
                start_v = v
        
        if start_v == -1:
            break
            
        # Greedily extend path from start_v
        path = [start_v]
        visited_in_path = {start_v}
        current = start_v
        
        while True:
            # Find next vertex
            # Heuristic: Pick neighbor with highest remaining degree
            next_v = -1
            max_next_deg = -1
            for neighbor in rem_adj[current]:
                if neighbor not in visited_in_path:
                    if len(rem_adj[neighbor]) > max_next_deg:
                        max_next_deg = len(rem_adj[neighbor])
                        next_v = neighbor
            
            if next_v == -1:
                break
            
            # Commit to edge (current, next_v)
            path.append(next_v)
            visited_in_path.add(next_v)
            
            # Remove edge from structures
            e = (min(current, next_v), max(current, next_v))
            remaining_edges.discard(e)
            rem_adj[current].discard(next_v)
            rem_adj[next_v].discard(current)
            
            current = next_v
            
            # Optional: Stop if we've used many edges to prevent very long loops? 
            # No, simple path constraint prevents infinite loops.
            # But we should stop if current has no unvisited neighbors.
            if not any(nb not in visited_in_path for nb in rem_adj[current]):
                break
                
        if len(path) > 1:
            paths.append(path)
            
    # Verify coverage
    covered = set()
    valid = True
    for path in paths:
        if len(path) < 2:
            continue
        if len(set(path)) != len(path):
            valid = False
            break
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            e = (min(a, b), max(a, b))
            if e in covered or e not in set((min(u,v), max(u,v)) for u,v in edges):
                valid = False
                break
            covered.add(e)
        if not valid:
            break
            
    if not valid or len(covered) != len(set((min(u,v), max(u,v)) for u,v in edges)):
        # Fallback: Single edges (always valid, just not minimal)
        paths = [[u, v] for (u, v) in edges]
        
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time left, use fallback
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            # Allocate time budget
            remaining_time = deadline - elapsed - 5
            time_budget = max(1.0, min(20.0, remaining_time / 3))
            paths = _solve_graph(*spec, time_budget=time_budget)
            results.append(paths)
    
    return results
