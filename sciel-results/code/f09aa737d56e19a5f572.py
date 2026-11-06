import time
from collections import defaultdict, deque
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def find_longest_path(adj, remaining_edges, step_limit=50000):
        """Find a long simple path in the graph defined by remaining_edges."""
        if not remaining_edges:
            return []
        
        # Build degree map for remaining edges
        deg = defaultdict(int)
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
        
        # Start from a node with highest degree
        if not deg:
            return []
        start = max(deg.keys(), key=lambda x: deg[x])
        
        best_path = []
        best_len = 0
        steps = 0
        
        # DFS with pruning
        def dfs(current, path, visited, used_edges):
            nonlocal best_path, best_len, steps
            if steps > step_limit:
                return
            steps += 1
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            # Try to extend
            for neighbor in adj[current]:
                if neighbor in visited:
                    continue
                # Check if edge (current, neighbor) is in remaining_edges
                e = tuple(sorted((current, neighbor)))
                if e not in remaining_edges:
                    continue
                if e in used_edges:
                    continue
                visited.add(neighbor)
                used_edges.add(e)
                path.append(neighbor)
                dfs(neighbor, path, visited, used_edges)
                path.pop()
                visited.remove(neighbor)
                used_edges.remove(e)
                if steps > step_limit:
                    return
        
        dfs(start, [start], {start}, set())
        
        # If best_path is just one node, try other starts
        if best_len <= 1:
            for s in deg:
                if len(best_path) > 1:
                    break
                steps = 0
                best_path = []
                best_len = 0
                dfs(s, [s], {s}, set())
        
        return best_path
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > DEADLINE:
            # Return best effort: each edge as a path
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        if not edges:
            results.append([])
            continue
        
        adj = get_adjacency(edges)
        remaining = set(edges)
        paths = []
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                # Fallback: each remaining edge as a path
                for e in remaining:
                    paths.append(list(e))
                remaining = set()
                break
            
            path = find_longest_path(adj, remaining)
            if not path or len(path) < 2:
                # Single node or empty; take one edge
                e = next(iter(remaining))
                paths.append(list(e))
                remaining.discard(e)
                continue
            
            # Extract edges from path
            path_edges = set()
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                path_edges.add(e)
            
            # Remove these edges from remaining
            remaining -= path_edges
            paths.append(path)
        
        results.append(paths)
    
    return results
