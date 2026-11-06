import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 3:
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(*spec)
        paths = _solve_instance(edges, deadline)
        results.append(paths)
        
    return results

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve_instance(edges, deadline):
    if not edges:
        return []
    
    # Multiple random restarts with greedy + optimal merge
    best_paths = None
    best_count = float('inf')
    
    num_restarts = 8
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        random.seed(restart * 12345 + 678)
        
        # Greedy decomposition with randomized choices
        paths = _greedy_decompose(edges, deadline)
        
        # Optimal merge via maximum matching
        paths = _optimal_merge(paths, edges, deadline)
        
        # Local search: try to split and re-merge
        paths = _local_search(paths, edges, deadline)
        
        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = [list(p) for p in paths]
    
    return best_paths if best_paths is not None else _greedy_decompose(edges, deadline)

def _greedy_decompose(edges, deadline):
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    paths = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Pick start vertex: prefer odd degree, then high degree
        candidates = [v for v in adj if adj[v]]
        if not candidates:
            break
            
        odd_verts = [v for v in candidates if degree.get(v, 0) % 2 == 1]
        if odd_verts:
            # Random choice among odd vertices with some bias
            weights = [degree[v] for v in odd_verts]
            total = sum(weights)
            r = random.random() * total
            cum = 0
            start = odd_verts[0]
            for v, w in zip(odd_verts, weights):
                cum += w
                if r <= cum:
                    start = v
                    break
        else:
            # Pick highest degree vertex with some randomness
            candidates.sort(key=lambda x: (-degree.get(x, 0), random.random()))
            start = candidates[0]
            
        # Find long path using Warnsdorff with randomization
        path = [start]
        visited = {start}
        current = start
        
        while True:
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            
            # Warnsdorff: prefer neighbors with fewer unvisited neighbors
            scored = []
            for n in neighbors:
                unvisited_count = sum(1 for y in adj[n] if y not in visited)
                # Add small random perturbation
                scored.append((unvisited_count + random.random() * 0.5, n))
            scored.sort()
            next_v = scored[0][1]
            
            path.append(next_v)
            visited.add(next_v)
            
            # Remove edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            adj[current].discard(next_v)
            adj[next_v].discard(current)
            degree[current] = degree.get(current, 0) - 1
            degree[next_v] = degree.get(next_v, 0) - 1
            
            if not adj[current]:
                del adj[current]
            if not adj[next_v]:
                del adj[next_v]
            
            current = next_v
            
        paths.append(path)
        
    return paths

def _optimal_merge(paths, edges, deadline):
    """Merge paths using maximum bipartite matching on endpoint compatibility."""
    if len(paths) <= 1:
        return paths
    
    # Build compatibility graph
    # Node i represents path i
    # We want to find maximum set of merges
    # Two paths can be merged if they share an endpoint and are vertex-disjoint otherwise
    
    n_paths = len(paths)
    # For each path, store its vertices as a set for quick lookup
    path_sets = [set(p) for p in paths]
    
    # Build merge compatibility matrix
    # merge[i][j] = True if path i and path j can be merged
    # We need to track direction: which end connects to which
    
    # For matching, we create a bipartite graph:
    # Left side: path endpoints (each path has 2 endpoints)
    # Right side: path endpoints
    # Edge exists if endpoints match and paths are compatible
    
    # Actually, simpler: greedily find maximum matching of merges
    # Use a simple greedy: repeatedly find any valid merge and do it
    
    merged = True
    while merged:
        merged = False
        best_merge = None
        best_i = -1
        best_j = -1
        best_type = None  # 'end_start', 'end_end', 'start_start', 'start_end'
        
        for i in range(len(paths)):
            if time.time() > deadline:
                return paths
            for j in range(i + 1, len(paths)):
                pi = paths[i]
                pj = paths[j]
                
                # Check all 4 merge types
                for merge_type in ['end_start', 'end_end', 'start_start', 'start_end']:
                    if merge_type == 'end_start':
                        if pi[-1] == pj[0]:
                            # Check vertex disjointness (except shared endpoint)
                            set_i = set(pi[:-1])
                            set_j = set(pj)
                            if not set_i & set_j:
                                if best_merge is None:
                                    best_merge = pi + pj[1:]
                                    best_i, best_j, best_type = i, j, merge_type
                    elif merge_type == 'end_end':
                        if pi[-1] == pj[-1]:
                            rev_pj = pj[::-1]
                            set_i = set(pi[:-1])
                            set_j = set(rev_pj)
                            if not set_i & set_j:
                                if best_merge is None:
                                    best_merge = pi + rev_pj[1:]
                                    best_i, best_j, best_type = i, j, merge_type
                    elif merge_type == 'start_start':
                        if pi[0] == pj[0]:
                            rev_pi = pi[::-1]
                            set_i = set(rev_pi[:-1])
                            set_j = set(pj)
                            if not set_i & set_j:
                                if best_merge is None:
                                    best_merge = rev_pi + pj[1:]
                                    best_i, best_j, best_type = i, j, merge_type
                    elif merge_type == 'start_end':
                        if pi[0] == pj[-1]:
                            rev_pj = pj[::-1]
                            set_i = set(pi)
                            set_j = set(rev_pj[:-1])
                            if not set_i & set_j:
                                if best_merge is None:
                                    best_merge = rev_pj + pi[1:]
                                    best_i, best_j, best_type = i, j, merge_type
        
        if best_merge is not None:
            # Remove paths i and j, add merged path
            new_paths = []
            for idx in range(len(paths)):
                if idx != best_i and idx != best_j:
                    new_paths.append(paths[idx])
            new_paths.append(best_merge)
            paths = new_paths
            merged = True
    
    return paths

def _local_search(paths, edges, deadline):
    """Try to improve by splitting long paths and re-merging differently."""
    if len(paths) <= 2:
        return paths
    
    best_paths = [list(p) for p in paths]
    best_count = len(best_paths)
    
    max_iters = 50
    for it in range(max_iters):
        if time.time() > deadline - 1:
            break
        if best_count <= 1:
            break
        
        # Try to split a path at a random internal point
        if len(paths) < 2:
            break
            
        path_idx = random.randint(0, len(paths) - 1)
        path = paths[path_idx]
        
        if len(path) < 3:
            continue
        
        # Pick a split point
        split_pos = random.randint(1, len(path) - 2)
        
        part1 = path[:split_pos + 1]
        part2 = path[split_pos:]
        
        # Check both parts are simple paths using valid edges
        if not _is_valid_path(part1, edges) or not _is_valid_path(part2, edges):
            continue
        
        # Create new path set
        new_paths = []
        for idx, p in enumerate(paths):
            if idx != path_idx:
                new_paths.append(list(p))
        new_paths.append(part1)
        new_paths.append(part2)
        
        # Try to optimally merge the new set
        merged = _optimal_merge(new_paths, edges, deadline)
        
        if len(merged) < best_count:
            best_count = len(merged)
            best_paths = [list(p) for p in merged]
            paths = merged
        else:
            # Keep original
            continue
    
    return best_paths

def _is_valid_path(path, edges):
    """Check if path is simple and uses only valid edges."""
    if len(path) < 2:
        return len(path) == 1
    
    # Check simplicity
    if len(set(path)) != len(path):
        return False
    
    # Check edges
    edge_set = set(tuple(sorted(e)) for e in edges)
    for i in range(len(path) - 1):
        e = tuple(sorted((path[i], path[i + 1])))
        if e not in edge_set:
            return False
    
    return True
