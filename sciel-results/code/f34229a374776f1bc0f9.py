"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_path_cover(n, edges):
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        remaining = set(edge_set)
        paths = []
        
        while remaining:
            # Build adjacency for remaining edges
            adj = {i: set() for i in range(n)}
            for (u, v) in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            # Choose start: vertex with highest degree in remaining graph
            start = max(range(n), key=lambda v: len(adj[v]))
            
            # Greedy path extension
            path = [start]
            visited = {start}
            
            while True:
                curr = path[-1]
                # Find unvisited neighbors
                candidates = [nb for nb in adj[curr] if nb not in visited]
                if not candidates:
                    break
                # Choose neighbor with fewest options (to avoid getting stuck later)
                next_v = min(candidates, key=lambda x: len(adj[x]))
                path.append(next_v)
                visited.add(next_v)
            
            # Remove edges used in this path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = find_path_cover(n, edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END

