import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_paths(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        # Build adjacency with edge sets for fast removal
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            e_key = (min(u, v), max(u, v))
            adj[u].add((v, e_key))
            adj[v].add((u, e_key))
            
        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
        paths = []
        
        while remaining_edges:
            # Calculate degrees in remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Find start vertex: Max degree
            start_candidates = [i for i in range(n) if deg[i] > 0]
            if not start_candidates:
                break
            start = max(start_candidates, key=lambda x: deg[x])
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Get available neighbors using Warnsdorff's rule
                candidates = []
                for v, e_key in adj[current]:
                    if v in visited:
                        continue
                    if e_key in remaining_edges:
                        # Count available neighbors of v (Warnsdorff: prefer fewer options)
                        avail_count = 0
                        for w, ew in adj[v]:
                            if w not in visited:
                                if ew in remaining_edges:
                                    avail_count += 1
                        candidates.append((avail_count, v, e_key))
                
                if not candidates:
                    break
                
                # Sort by available count (ascending) - Warnsdorff's rule
                candidates.sort(key=lambda x: x[0])
                _, next_v, e_to_remove = candidates[0]
                
                remaining_edges.remove(e_to_remove)
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths
    
    def try_decomposition(edges, n, max_restarts=5):
        """Try multiple times with different heuristics and return best."""
        best_paths = None
        best_count = float('inf')
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            if attempt == 0:
                # Original Warnsdorff
                paths = decompose_paths(edges)
            elif attempt == 1:
                # Try with max-degree neighbor preference
                paths = _decompose_max_degree(edges, n)
            elif attempt == 2:
                # Try with mixed heuristic
                paths = _decompose_mixed(edges, n)
            elif attempt == 3:
                # Try random restarts
                paths = _decompose_randomized(edges, n, seed=attempt)
            else:
                paths = _decompose_warnsdorff_v2(edges, n)
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
        
        return best_paths if best_paths else decompose_paths(edges)
    
    def _decompose_max_degree(edges, n):
        """Prefer neighbor with highest remaining degree."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            e_key = (min(u, v), max(u, v))
            adj[u].add((v, e_key))
            adj[v].add((u, e_key))
        
        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
        paths = []
        
        while remaining_edges:
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            start_candidates = [i for i in range(n) if deg[i] > 0]
            if not start_candidates:
                break
            start = max(start_candidates, key=lambda x: deg[x])
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                candidates = []
                for v, e_key in adj[current]:
                    if v in visited:
                        continue
                    if e_key in remaining_edges:
                        candidates.append((deg[v], v, e_key))
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, next_v, e_to_remove = candidates[0]
                
                remaining_edges.remove(e_to_remove)
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths
    
    def _decompose_mixed(edges, n):
        """Mixed heuristic: 70% Warnsdorff, 30% max degree."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            e_key = (min(u, v), max(u, v))
            adj[u].add((v, e_key))
            adj[v].add((u, e_key))
        
        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
        paths = []
        rng = random.Random(42)
        
        while remaining_edges:
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            start_candidates = [i for i in range(n) if deg[i] > 0]
            if not start_candidates:
                break
            start = max(start_candidates, key=lambda x: deg[x])
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                candidates = []
                for v, e_key in adj[current]:
                    if v in visited:
                        continue
                    if e_key in remaining_edges:
                        avail_count = 0
                        for w, ew in adj[v]:
                            if w not in visited:
                                if ew in remaining_edges:
                                    avail_count += 1
                        # Mixed score: lower is better
                        score = 0.7 * avail_count + 0.3 * (-deg[v])
                        candidates.append((score, v, e_key))
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: x[0])
                _, next_v, e_to_remove = candidates[0]
                
                remaining_edges.remove(e_to_remove)
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths
    
    def _decompose_randomized(edges, n, seed=0):
        """Randomized heuristic for diversity."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            e_key = (min(u, v), max(u, v))
            adj[u].add((v, e_key))
            adj[v].add((u, e_key))
        
        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
        paths = []
        rng = random.Random(seed)
        
        while remaining_edges:
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            start_candidates = [i for i in range(n) if deg[i] > 0]
            if not start_candidates:
                break
            # Randomly pick from top 3 degree vertices
            start_candidates.sort(key=lambda x: deg[x], reverse=True)
            top_k = min(3, len(start_candidates))
            start = rng.choice(start_candidates[:top_k])
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                candidates = []
                for v, e_key in adj[current]:
                    if v in visited:
                        continue
                    if e_key in remaining_edges:
                        avail_count = 0
                        for w, ew in adj[v]:
                            if w not in visited:
                                if ew in remaining_edges:
                                    avail_count += 1
                        score = rng.random() * 0.5 + avail_count * 0.3 - deg[v] * 0.2
                        candidates.append((score, v, e_key))
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: x[0])
                _, next_v, e_to_remove = candidates[0]
                
                remaining_edges.remove(e_to_remove)
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths
    
    def _decompose_warnsdorff_v2(edges, n):
        """Improved Warnsdorff with lookahead."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            e_key = (min(u, v), max(u, v))
            adj[u].add((v, e_key))
            adj[v].add((u, e_key))
        
        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
        paths = []
        
        while remaining_edges:
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            start_candidates = [i for i in range(n) if deg[i] > 0]
            if not start_candidates:
                break
            # Prefer vertices with degree 1 or 2 (endpoints)
            low_deg = [v for v in start_candidates if deg[v] <= 2]
            if low_deg:
                start = low_deg[0]
            else:
                start = max(start_candidates, key=lambda x: deg[x])
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                candidates = []
                for v, e_key in adj[current]:
                    if v in visited:
                        continue
                    if e_key in remaining_edges:
                        avail_count = 0
                        for w, ew in adj[v]:
                            if w not in visited:
                                if ew in remaining_edges:
                                    avail_count += 1
                        # Penalize if v would become isolated
                        penalty = 0
                        if avail_count == 1 and deg[v] == 2:
                            penalty = 10
                        score = avail_count + penalty
                        candidates.append((score, v, e_key))
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: x[0])
                _, next_v, e_to_remove = candidates[0]
                
                remaining_edges.remove(e_to_remove)
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths
    
    def validate_and_fix(paths, edges, n):
        """Validate paths and fix any issues."""
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        used_edges = set()
        valid_paths = []
        
        for path in paths:
            if len(path) < 2:
                continue
            valid = True
            path_edges = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e_key = (min(u, v), max(u, v))
                if e_key not in edge_set:
                    valid = False
                    break
                if e_key in used_edges:
                    valid = False
                    break
                path_edges.append(e_key)
            
            if valid:
                used_edges.update(path_edges)
                valid_paths.append(path)
            else:
                # Fallback: add each edge as separate path
                for e in path_edges:
                    if e not in used_edges:
                        used_edges.add(e)
                        valid_paths.append([e[0], e[1]])
        
        # Add any missing edges as single-edge paths
        for e in edge_set:
            if e not in used_edges:
                used_edges.add(e)
                valid_paths.append([e[0], e[1]])
        
        return valid_paths
    
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result (each edge a path) for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        n = max(max(e) for e in edges) + 1 if edges else 0
        
        # Try multiple decomposition strategies
        paths = try_decomposition(edges, n, max_restarts=5)
        
        # Validate and fix
        paths = validate_and_fix(paths, edges, n)
        
        results.append(paths)
    
    return results
