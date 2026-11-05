import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    # Deadline guard: 150s limit, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 138.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_greedy(edges, rng, strategy='random'):
        """
        Runs a greedy path decomposition algorithm.
        Tries to build long paths by extending from both ends.
        Uses randomization in tie-breaking to explore different solutions.
        """
        if not edges:
            return []
        
        # Copy edges to a mutable set for removal
        remaining = set(edges)
        
        paths = []
        
        # Build adjacency for fast neighbor lookup
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        # Helper to get neighbors of a node in the remaining graph
        def get_neighbors(u):
            return list(adj[u])

        # Helper to remove edge
        def remove_edge(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            remaining.discard((u, v))
            remaining.discard((v, u))

        # Helper to extend path
        def extend_path(path):
            # path is a list of vertices
            # Try to extend from both ends until no more extension is possible
            changed = True
            while changed:
                changed = False
                # Try extending the back end
                if len(path) > 0:
                    back = path[-1]
                    neighbors = get_neighbors(back)
                    # Filter neighbors that are not already in the path
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            # Prefer neighbors with lower remaining degree
                            degs = {}
                            for n in valid_neighbors:
                                degs[n] = len(adj[n])
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.append(next_node)
                        remove_edge(back, next_node)
                        changed = True
                
                # Try extending the front end
                if len(path) > 0:
                    front = path[0]
                    neighbors = get_neighbors(front)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            degs = {}
                            for n in valid_neighbors:
                                degs[n] = len(adj[n])
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True
                        
        # Main loop: construct paths
        while remaining:
            # Pick a starting edge
            # Heuristic: Pick an edge incident to a vertex with the highest degree
            # Calculate degrees
            degrees = defaultdict(int)
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            
            # Find vertex with max degree
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            start_u = rng.choice(candidates)
            
            # Pick a neighbor for the starting edge
            neighbors = get_neighbors(start_u)
            if not neighbors:
                continue 
            
            # Prefer neighbors with higher degree to create longer paths
            if strategy == 'low_deg':
                # For starting edge, prefer HIGH degree neighbors
                degs = {n: len(adj[n]) for n in neighbors}
                max_n_deg = max(degs.values())
                high_cands = [n for n in neighbors if degs[n] == max_n_deg]
                start_v = rng.choice(high_cands)
            else:
                start_v = rng.choice(neighbors)
            
            # Initialize path
            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            
            # Extend path
            extend_path(path)
            
            paths.append(path)
            
            # Check deadline
            if time.time() > deadline:
                break

        return paths

    def decompose_greedy_deterministic(edges, start_vertex=None):
        """
        Deterministic greedy decomposition.
        Starts from a specific vertex if provided, otherwise picks highest degree.
        Always picks the first valid neighbor in sorted order.
        """
        if not edges:
            return []
        
        remaining = set(edges)
        paths = []
        
        # Build adjacency for fast neighbor lookup
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        def get_neighbors(u):
            return list(adj[u])

        def remove_edge(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            remaining.discard((u, v))
            remaining.discard((v, u))

        def extend_path(path):
            changed = True
            while changed:
                changed = False
                if len(path) > 0:
                    back = path[-1]
                    neighbors = sorted(get_neighbors(back))
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        next_node = valid_neighbors[0]
                        path.append(next_node)
                        remove_edge(back, next_node)
                        changed = True
                
                if len(path) > 0:
                    front = path[0]
                    neighbors = sorted(get_neighbors(front))
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        next_node = valid_neighbors[0]
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True
                        
        while remaining:
            if start_vertex is not None and start_vertex in [u for u, v in remaining] + [v for u, v in remaining]:
                start_u = start_vertex
                start_vertex = None  # Only use once
            else:
                degrees = defaultdict(int)
                for u, v in remaining:
                    degrees[u] += 1
                    degrees[v] += 1
                max_deg = max(degrees.values())
                candidates = sorted([v for v, d in degrees.items() if d == max_deg])
                start_u = candidates[0]
            
            neighbors = sorted(get_neighbors(start_u))
            if not neighbors:
                continue 
            
            start_v = neighbors[0]
            
            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            
            extend_path(path)
            
            paths.append(path)
            
            if time.time() > deadline:
                break

        return paths

    def try_merge_paths(paths, edges):
        """
        Try to merge two paths by finding a connecting path in the remaining edges.
        Returns (merged_paths, success) where merged_paths is the new list if successful.
        """
        if len(paths) < 2:
            return paths, False
        
        # Build set of all edges used in paths
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                used_edges.add(tuple(sorted((u, v))))
        
        # Remaining edges not in any path
        remaining_edges = set(edges) - used_edges
        if not remaining_edges:
            return paths, False
        
        # Build adjacency for remaining edges
        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each pair of paths, check if we can connect them
        # Path A ends at a, Path B starts at b: can we go a -> ... -> b using remaining edges?
        # Or Path A ends at a, Path B ends at b: can we go a -> ... -> b and reverse B?
        
        # We'll try to find a simple path in remaining graph between endpoints
        # This is expensive, so we'll limit the search
        
        n_paths = len(paths)
        best_new_paths = None
        best_len = n_paths
        
        # Try all pairs of paths (limited to avoid O(n^2 * BFS) being too slow)
        # Only try if n_paths is small enough
        if n_paths > 15:
            return paths, False
        
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                path_i = paths[i]
                path_j = paths[j]
                
                # Get endpoints
                start_i, end_i = path_i[0], path_i[-1]
                start_j, end_j = path_j[0], path_j[-1]
                
                # Try connecting end_i to start_j
                # Find a path from end_i to start_j in remaining graph
                def bfs_path(src, dst, adj, max_depth=5):
                    """Find a simple path from src to dst with depth limit."""
                    if src == dst:
                        return [src]
                    visited = {src}
                    queue = [(src, [src])]
                    while queue:
                        node, path = queue.pop(0)
                        if len(path) > max_depth:
                            continue
                        for nb in adj.get(node, []):
                            if nb == dst:
                                return path + [nb]
                            if nb not in visited:
                                visited.add(nb)
                                queue.append((nb, path + [nb]))
                    return None
                
                # Try end_i -> start_j
                connector = bfs_path(end_i, start_j, adj)
                if connector:
                    # Merge: path_i + connector[1:] + path_j
                    new_path = path_i + connector[1:]
                    # Check simplicity
                    if len(set(new_path)) == len(new_path):
                        new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                        new_paths.append(new_path)
                        if len(new_paths) < best_len:
                            best_len = len(new_paths)
                            best_new_paths = new_paths
                            return new_paths, True
                
                # Try start_i -> end_j (reverse path_j)
                connector = bfs_path(start_i, end_j, adj)
                if connector:
                    # Merge: reverse(path_j) + connector[1:] + path_i
                    rev_j = list(reversed(path_j))
                    new_path = rev_j + connector[1:] + path_i
                    if len(set(new_path)) == len(new_path):
                        new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                        new_paths.append(new_path)
                        if len(new_paths) < best_len:
                            best_len = len(new_paths)
                            best_new_paths = new_paths
                            return new_paths, True
                
                # Try end_i -> end_j (reverse path_j)
                connector = bfs_path(end_i, end_j, adj)
                if connector:
                    # Merge: path_i + connector[1:] + reverse(path_j)
                    rev_j = list(reversed(path_j))
                    new_path = path_i + connector[1:] + rev_j
                    if len(set(new_path)) == len(new_path):
                        new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                        new_paths.append(new_path)
                        if len(new_paths) < best_len:
                            best_len = len(new_paths)
                            best_new_paths = new_paths
                            return new_paths, True
                
                # Try start_i -> start_j
                connector = bfs_path(start_i, start_j, adj)
                if connector:
                    # Merge: reverse(path_j) + connector[1:] + path_i  -- wait, this doesn't work directly
                    # Actually: path_i starts at start_i, path_j starts at start_j
                    # If we connect start_i to start_j, we can do: reverse(path_j) from start_j... 
                    # No, let's think: we want to merge into one path.
                    # Option: path_i is start_i->...->end_i, path_j is start_j->...->end_j
                    # Connect start_i to start_j: then we can do start_j->...->end_j then jump? No.
                    # Better: connect end_i to start_j or end_i to end_j etc.
                    # Let's skip this case for now.
                    pass
        
        return paths, False

    def local_improve(paths, edges, max_rounds=5):
        """
        Repeatedly try to merge paths to reduce the count.
        """
        current_paths = list(paths)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            new_paths, success = try_merge_paths(current_paths, edges)
            if not success:
                break
            current_paths = new_paths
        return current_paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        # Time allocation: run as many iterations as possible
        # Each iteration is very fast for these small graphs
        
        # First, try deterministic strategies
        # 1. Deterministic greedy from highest degree
        if time.time() < deadline - 1:
            paths = decompose_greedy_deterministic(edges)
            if validate_paths(paths, edges):
                paths = local_improve(paths, edges)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
        
        # 2. Deterministic greedy from each vertex (as starting point)
        vertices = sorted(set(u for e in edges for u in e))
        for v in vertices:
            if time.time() > deadline - 1:
                break
            paths = decompose_greedy_deterministic(edges, start_vertex=v)
            if validate_paths(paths, edges):
                paths = local_improve(paths, edges)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
        
        # 3. Randomized strategies
        iterations = 5000
        for i in range(iterations):
            if time.time() > deadline - 1:
                break
            
            # Alternate between random and low-degree heuristics
            if i % 3 == 0:
                strategy = 'random'
            elif i % 3 == 1:
                strategy = 'low_deg'
            else:
                strategy = 'random'
            
            rng = random.Random(seed * 10000 + i)
            paths = decompose_greedy(edges, rng, strategy)
            
            if validate_paths(paths, edges):
                paths = local_improve(paths, edges)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
                    
                    # Early termination if we hit a very good score
                    # Gallai's bound for n=20 is 10, n=25 is 13, n=30 is 15
                    # But we might do better. If we get close to lower bounds, stop.
                    if best_count <= 5:
                        break

        if best_paths is None:
            # Fallback: return single edges if greedy failed completely (shouldn't happen)
            return [[list(e)] for e in edges]
            
        return best_paths

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            # Check simple path
            if len(set(path)) != len(path):
                return False
            
            # Check edges exist and are not reused
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        if used_edges != edge_set:
            return False
        
        return True

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    # Allocate time per instance based on expected difficulty
    # Give more time to later instances if we're ahead of schedule
    time_per_instance = 40.0  # ~120s total for 3 instances
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        # Adjust deadline for this instance
        instance_deadline = start_time + (idx + 1) * time_per_instance
        if instance_deadline > deadline:
            instance_deadline = deadline
        
        # Temporarily adjust the global deadline for this instance
        old_deadline = deadline
        deadline = instance_deadline
        
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
        deadline = old_deadline
        
        if time.time() > deadline:
            break
            
    # If we didn't finish all specs, ensure we return a valid structure
    while len(results) < 3:
        # Generate a trivial valid solution: each edge as its own path
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    # Quick test
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
