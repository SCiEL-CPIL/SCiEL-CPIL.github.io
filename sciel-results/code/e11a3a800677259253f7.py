"""Path decomposition of graphs using greedy + randomized restarts with full merge heuristics."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    best_results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = _instance(*spec)
            best_results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        result = _decompose(edges, deadline)
        if result is not None:
            best_results.append(result)
        else:
            # Fallback: each edge as its own path
            best_results.append([list(e) for e in edges])
    
    return best_results


def _instance(n, p, seed):
    """Generate the instance graph exactly as in the verifier."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    """Decompose edges into minimum number of edge-disjoint simple paths."""
    if not edges:
        return []
    
    best_paths = None
    best_count = len(edges)  # Worst case: each edge is its own path
    
    # Number of restarts based on available time
    # Estimate time per restart: ~0.1s? 140s / 0.1 = 1400 restarts. Cap it.
    num_restarts = max(100, int((deadline - time.time()) * 10))
    num_restarts = min(num_restarts, 500)
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        paths = _greedy_decompose(edges, rng, deadline)
        
        if paths is not None:
            # Post-process: aggressive merge
            paths = _merge_paths_full(paths, edges)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                # Early stop if we hit theoretical lower bound
                lower_bound = _lower_bound(edges)
                if count <= lower_bound:
                    break
    
    if best_paths is None:
        return [list(e) for e in edges]
    
    return best_paths


def _lower_bound(edges):
    """Compute theoretical lower bound on number of paths."""
    if not edges:
        return 0
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    odd_count = sum(1 for d in degree.values() if d % 2 == 1)
    return max(1, (odd_count + 1) // 2)


def _greedy_decompose(edges, rng, deadline):
    """
    Greedy path decomposition:
    1. Build adjacency from remaining edges
    2. Repeatedly find a long simple path using remaining edges
    3. Remove those edges and repeat
    """
    remaining = set(edges)
    if not remaining:
        return []
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    degree = defaultdict(int)
    for u, v in remaining:
        degree[u] += 1
        degree[v] += 1
    
    paths = []
    
    while remaining:
        if time.time() > deadline:
            return None
        
        # Find a good starting vertex
        start = _pick_start(adj, degree, rng)
        if start is None:
            # Fallback: pick any vertex with edges
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
            start = rng.choice(candidates)
        
        # Find a long simple path starting from start
        path = _find_long_path(start, adj, degree, rng, remaining, deadline)
        
        if path is None or len(path) < 2:
            # Can't find a path; take a single edge
            if adj[start]:
                next_v = rng.choice(list(adj[start]))
                path = [start, next_v]
            else:
                # Pick any remaining edge
                e = next(iter(remaining))
                path = [e[0], e[1]]
        
        # Extract edges from path and remove them
        path_edges = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                path_edges.add(edge)
        
        if not path_edges:
            # Shouldn't happen, but safety
            break
        
        for e in path_edges:
            remaining.discard(e)
        
        # Update adjacency and degrees
        for u, v in path_edges:
            adj[u].discard(v)
            adj[v].discard(u)
            degree[u] -= 1
            degree[v] -= 1
        
        # Clean up empty adj entries
        for v in list(adj.keys()):
            if not adj[v]:
                del adj[v]
        
        paths.append(path)
    
    return paths


def _pick_start(adj, degree, rng):
    """Pick a starting vertex, preferring odd-degree vertices."""
    odd_vertices = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
    if odd_vertices:
        return rng.choice(odd_vertices)
    
    # Prefer high-degree vertices to keep the graph connected longer?
    # Or low degree to finish off leaves?
    # Usually, starting at high degree is better for long paths.
    candidates = [v for v in adj if adj[v]]
    if not candidates:
        return None
    
    # Weight by degree
    weights = [max(1, degree.get(v, 0)) for v in candidates]
    total = sum(weights)
    r = rng.random() * total
    cum = 0
    for v, w in zip(candidates, weights):
        cum += w
        if r <= cum:
            return v
    return candidates[-1]


def _find_long_path(start, adj, degree, rng, remaining, deadline):
    """
    Find a long simple path starting from start using DFS with backtracking.
    """
    best_path = [start]
    
    # Use a greedy approach with limited backtracking
    _greedy_extend(start, adj, best_path, rng, deadline)
    
    return best_path if len(best_path) >= 2 else None


def _greedy_extend(start, adj, best_path, rng, deadline):
    """
    Greedy path extension.
    """
    path = [start]
    visited = {start}
    
    max_depth = 100
    for step in range(max_depth):
        if time.time() > deadline:
            break
        current = path[-1]
        neighbors = [nb for nb in adj[current] if nb not in visited]
        
        if not neighbors:
            break
        
        # Choose next vertex: Warnsdorff's rule
        # Prefer neighbors with fewer unvisited neighbors
        def score(nb):
            unvisited_count = sum(1 for x in adj[nb] if x not in visited)
            return unvisited_count
        
        # Sometimes pick greedily, sometimes randomly
        if rng.random() < 0.85:
            neighbors.sort(key=score)
            # Pick from top 2-3
            top_k = min(3, len(neighbors))
            choice = rng.choice(neighbors[:top_k])
        else:
            choice = rng.choice(neighbors)
        
        path.append(choice)
        visited.add(choice)
        
        if len(path) > len(best_path):
            best_path[:] = path[:]
    
    return path


def _merge_paths_full(paths, edges):
    """
    Merge paths that can be concatenated, considering reversals.
    """
    if len(paths) <= 1:
        return paths
    
    # Build edge set for validation
    edge_set = set(edges)
    
    merged = True
    while merged:
        merged = False
        new_paths = []
        used_indices = set()
        
        for i in range(len(paths)):
            if i in used_indices:
                continue
                
            current_path = paths[i]
            
            for j in range(len(paths)):
                if j == i or j in used_indices:
                    continue
                
                other_path = paths[j]
                
                # Try 4 combinations
                merged_path = None
                
                # 1. current end == other start
                if current_path[-1] == other_path[0]:
                    candidate = current_path + other_path[1:]
                    if _is_valid_simple_path(candidate, edge_set):
                        merged_path = candidate
                
                # 2. current end == other end (reverse other)
                if current_path[-1] == other_path[-1]:
                    rev_other = other_path[::-1]
                    candidate = current_path + rev_other[1:]
                    if _is_valid_simple_path(candidate, edge_set):
                        merged_path = candidate
                
                # 3. current start == other start (reverse current)
                if current_path[0] == other_path[0]:
                    rev_current = current_path[::-1]
                    candidate = rev_current + other_path[1:]
                    if _is_valid_simple_path(candidate, edge_set):
                        merged_path = candidate
                
                # 4. current start == other end (reverse other, merge other first? No, merge current first)
                # current starts at X, other ends at X.
                # Merge: other + current[1:]?
                # other ends at X. current starts at X.
                # So other + current[1:] is valid if we treat other as the prefix.
                # But we are iterating i as the "main" path.
                # If we merge, we can just add the result to new_paths.
                if current_path[0] == other_path[-1]:
                    candidate = other_path + current_path[1:]
                    if _is_valid_simple_path(candidate, edge_set):
                        merged_path = candidate
                
                if merged_path:
                    current_path = merged_path
                    used_indices.add(j)
                    merged = True
                    # Don't break, try to merge more into this path?
                    # Yes, keep looking for other paths to merge into current_path
                    # But we must be careful not to use j again.
                    # The loop continues for j.
        
        new_paths.append(current_path)
        
        paths = new_paths
        
        if len(paths) == 1:
            break
            
    return paths


def _is_valid_simple_path(path, edge_set):
    """Check if path is simple and uses valid edges."""
    if len(path) < 2:
        return True
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if tuple(sorted((u, v))) not in edge_set:
            return False
    return True

# EVOLVE-BLOCK-END
