import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_path_cover_v2(edges, adj, rng, temperature, strategy):
        """
        Greedy path cover that tries to maximize path length.
        strategy: 'leaf', 'min_deg', 'random'
        """
        remaining = set(map(tuple, edges))
        deg = defaultdict(int)
        for e in remaining:
            deg[e[0]] += 1
            deg[e[1]] += 1

        paths = []
        
        while remaining:
            if time.time() > deadline:
                for e in remaining:
                    paths.append(list(e))
                break

            # Choose start vertex based on strategy
            if strategy == 'leaf':
                leaves = [v for e in remaining for v in e if deg[v] == 1]
                if leaves:
                    start_v = rng.choice(leaves) if temperature > 0.01 else leaves[0]
                else:
                    min_deg = min(deg[v] for e in remaining for v in e)
                    candidates = [v for e in remaining for v in e if deg[v] == min_deg]
                    start_v = rng.choice(candidates) if temperature > 0.01 else candidates[0]
            elif strategy == 'min_deg':
                min_deg = min(deg[v] for e in remaining for v in e)
                candidates = [v for e in remaining for v in e if deg[v] == min_deg]
                start_v = rng.choice(candidates) if temperature > 0.01 else candidates[0]
            else:
                all_vertices = set()
                for e in remaining:
                    all_vertices.add(e[0])
                    all_vertices.add(e[1])
                start_v = rng.choice(list(all_vertices))

            # Build path from start_v
            path = [start_v]
            visited = {start_v}
            current = start_v
            
            while True:
                # Find available neighbors
                valid_neighbors = []
                for u in adj[current]:
                    edge = (min(current, u), max(current, u))
                    if edge in remaining and u not in visited:
                        valid_neighbors.append(u)
                
                if not valid_neighbors:
                    break
                
                # Choose next vertex
                if temperature <= 0.01:
                    # Deterministic: pick neighbor with fewest remaining degree
                    next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
                else:
                    # Weighted random
                    weights = [1.0 / (deg.get(u, 0) + 1.0) for u in valid_neighbors]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    next_v = valid_neighbors[-1]
                    for i, u in enumerate(valid_neighbors):
                        cum += weights[i]
                        if r <= cum:
                            next_v = u
                            break
                
                edge = (min(current, next_v), max(current, next_v))
                remaining.discard(edge)
                deg[current] -= 1
                deg[next_v] -= 1
                path.append(next_v)
                visited.add(next_v)
                current = next_v
            
            if len(path) > 1:
                paths.append(path)
            else:
                # Single vertex, try to extend
                for u in adj[start_v]:
                    edge = (min(start_v, u), max(start_v, u))
                    if edge in remaining:
                        path.append(u)
                        remaining.discard(edge)
                        deg[start_v] -= 1
                        deg[u] -= 1
                        break
                if len(path) > 1:
                    paths.append(path)
                else:
                    paths.append(path)
        
        # Handle remaining edges
        for e in remaining:
            paths.append(list(e))
        
        return paths

    def try_merge_paths(paths, edges, adj):
        """Try to merge paths using remaining edges between endpoints."""
        all_edges = set(map(tuple, edges))
        
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                used_edges.add((min(u, v), max(u, v)))
        
        remaining_edges = all_edges - used_edges
        
        improved = True
        max_iters = 300
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            for i in range(len(paths)):
                if improved:
                    break
                for j in range(len(paths)):
                    if i == j:
                        continue
                    
                    # Try: end of path i to start of path j
                    end_i = paths[i][-1]
                    start_j = paths[j][0]
                    edge = (min(end_i, start_j), max(end_i, start_j))
                    if edge in remaining_edges:
                        if start_j not in paths[i]:
                            paths[i] = paths[i] + paths[j]
                            remaining_edges.discard(edge)
                            paths.pop(j)
                            improved = True
                            break
                    
                    # Try: start of path i to end of path j
                    start_i = paths[i][0]
                    end_j = paths[j][-1]
                    edge = (min(start_i, end_j), max(start_i, end_j))
                    if edge in remaining_edges:
                        if end_j not in paths[i]:
                            paths[i] = list(reversed(paths[j])) + paths[i]
                            remaining_edges.discard(edge)
                            paths.pop(j)
                            improved = True
                            break
        
        return paths

    def validate_paths(paths, edges):
        """Validate that paths are valid and cover all edges exactly once."""
        all_edges = set(map(tuple, edges))
        used_edges = set()
        valid_paths = []
        
        for path in paths:
            if len(path) < 2:
                continue
            valid = True
            path_edge_set = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                if e not in all_edges or e in used_edges or e in path_edge_set:
                    valid = False
                    break
                path_edge_set.add(e)
            
            if valid and len(set(path)) == len(path):
                valid_paths.append(path)
                used_edges.update(path_edge_set)
        
        # Add remaining edges
        remaining = all_edges - used_edges
        for e in remaining:
            valid_paths.append(list(e))
        
        return valid_paths

    def solve_graph(edges, n):
        """Solve path cover for a single graph."""
        adj = build_adj(edges)
        best_cover = None
        best_count = len(edges)
        
        strategies = [
            (0.0, 'leaf'),
            (0.0, 'min_deg'),
            (0.1, 'leaf'),
            (0.2, 'leaf'),
            (0.3, 'leaf'),
            (0.4, 'min_deg'),
            (0.5, 'leaf'),
            (0.5, 'min_deg'),
            (0.6, 'leaf'),
            (0.7, 'min_deg'),
            (0.8, 'leaf'),
            (0.9, 'random'),
        ]
        
        for temp, mode in strategies:
            for trial in range(30):
                if time.time() > deadline:
                    break
                
                rng = random.Random(hash((temp, mode, trial, n)) % (2**32))
                cover = greedy_path_cover_v2(edges, adj, rng, temp, mode)
                cover = try_merge_paths(cover, edges, adj)
                cover = validate_paths(cover, edges)
                count = len(cover)
                
                if count < best_count:
                    best_count = count
                    best_cover = cover
                
                if best_count <= 8:
                    break
            
            if time.time() > deadline:
                break
            if best_count <= 8:
                break
        
        if best_cover is None:
            best_cover = [list(e) for e in edges]
        
        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        cover = solve_graph(edges, n)
        results.append(cover)

    return results
