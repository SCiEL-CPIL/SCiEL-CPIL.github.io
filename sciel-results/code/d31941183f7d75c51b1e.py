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
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
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

    def greedy_standard(edges, seed):
        """Standard greedy, no target count. Used as fallback."""
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        
        while remaining:
            # Pick highest degree vertex in remaining graph
            degs = {}
            for u, v in remaining:
                degs[u] = degs.get(u, 0) + 1
                degs[v] = degs.get(v, 0) + 1
            if not degs:
                break
            max_d = max(degs.values())
            cands = [v for v, d in degs.items() if d == max_d]
            start_u = rng.choice(cands)
            
            neighbors = [nb for nb in adj[start_u] if (start_u, nb) in remaining or (nb, start_u) in remaining]
            if not neighbors:
                # Isolated? Should not happen if remaining is not empty and connected components are handled
                # Just pick any edge
                e = next(iter(remaining))
                start_u, start_v = e
            else:
                start_v = rng.choice(neighbors)
                
            path = [start_u, start_v]
            if (start_u, start_v) in remaining:
                remaining.remove((start_u, start_v))
            else:
                remaining.remove((start_v, start_u))
                
            # Extend
            changed = True
            while changed:
                changed = False
                # Back
                back = path[-1]
                cands = []
                for nb in adj[back]:
                    if nb in path:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        cands.append(nb)
                if cands:
                    nb = rng.choice(cands)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    path.append(nb)
                    changed = True
                    continue
                # Front
                front = path[0]
                cands = []
                for nb in adj[front]:
                    if nb in path:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        cands.append(nb)
                if cands:
                    nb = rng.choice(cands)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    path.insert(0, nb)
                    changed = True
                    continue
                    
            paths.append(path)
            
        return paths

    def try_greedy_target(edges, k, seed_val):
        """
        Attempts to cover all edges with exactly k simple paths.
        Returns the list of paths if successful, else None.
        """
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None
            
        rng = random.Random(seed_val)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = [[] for _ in range(k)]
        
        # Pick k distinct edges to start k paths
        edge_list = list(remaining)
        rng.shuffle(edge_list)
        
        # We need to ensure we pick edges that don't immediately conflict in a way that makes it impossible?
        # Random selection is usually fine for small graphs.
        start_edges = edge_list[:k]
        
        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]
            
        # Extend paths
        # We will loop until no more extensions possible
        changed = True
        while changed:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)
            
            for i in idxs:
                if not paths[i]:
                    continue
                    
                # Try extending back
                back = paths[i][-1]
                candidates = []
                for nb in adj[back]:
                    if nb in paths[i]:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        candidates.append(nb)
                        
                if candidates:
                    # Heuristic: Pick neighbor with lowest degree in remaining?
                    # Or random? Random is simpler.
                    nb = rng.choice(candidates)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue
                    
                # Try extending front
                front = paths[i][0]
                candidates = []
                for nb in adj[front]:
                    if nb in paths[i]:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        candidates.append(nb)
                        
                if candidates:
                    nb = rng.choice(candidates)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue
                    
        if remaining:
            return None
        return paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency and degrees
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        
        # Theoretical Lower Bound
        if num_odd == 0:
            lb = 1 if len(edges) == 1 else 2
        else:
            lb = num_odd // 2
            
        best_paths = None
        best_count = float('inf')
        
        # Strategy 1: Greedy Target 'lb'
        # We try to construct exactly 'lb' paths.
        
        # Determine number of attempts based on time
        # ~30-40s per instance is reasonable.
        # try_greedy_target is O(E^2) roughly? No, O(E * V) for set lookups.
        # With E~100-200, it's very fast. 1000 attempts is <1s.
        # Let's do 5000 attempts per instance, capped by time.
        
        max_attempts = 5000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline - 5:
                break
            attempts += 1
            paths = try_greedy_target(edges, lb, seed * 10000 + attempts)
            if paths:
                # Validate
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
                    break # Found optimal (or near optimal) target count
        
        # If greedy target failed, fallback to standard greedy (minimizing count, no target)
        if best_paths is None:
            # Try a few standard greedy runs to find a good one
            for i in range(100):
                if time.time() > deadline - 5:
                    break
                paths = greedy_standard(edges, seed * 10000 + 10000 + i)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
            
            if best_paths is None:
                # Last resort: single edges
                best_paths = [[u, v] for u, v in edges]
                best_count = len(best_paths)

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        # Check time before starting instance
        if time.time() > deadline - 10:
            # Not enough time, use trivial decomposition for remaining
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
    # Fallback
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
