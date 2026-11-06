import time
from random import Random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_greedy(edges):
        if not edges:
            return []
        
        remaining_edges = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        
        def get_degree(v):
            return len(adj[v])
            
        def remove_edge(u, v):
            edge = (min(u, v), max(u, v))
            if edge in remaining_edges:
                remaining_edges.remove(edge)
            adj[u].discard(v)
            adj[v].discard(u)
            
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: add remaining edges as single-edge paths
                for u, v in remaining_edges:
                    paths.append([u, v])
                remaining_edges.clear()
                break
            
            # Find start vertex: min degree > 0 to clear leaves/low-degree nodes first
            candidates = [v for v in adj if get_degree(v) > 0]
            if not candidates:
                break
            
            start_v = min(candidates, key=lambda x: get_degree(x))
            
            path = [start_v]
            visited_in_path = {start_v}
            current = start_v
            
            while True:
                neighbors = list(adj[current])
                # Filter to unvisited vertices to ensure simple path (no cycles)
                valid_neighbors = [n for n in neighbors if n not in visited_in_path]
                
                if not valid_neighbors:
                    break
                    
                # Heuristic: Choose neighbor with LOWEST degree
                next_v = min(valid_neighbors, key=lambda x: get_degree(x))
                
                path.append(next_v)
                visited_in_path.add(next_v)
                remove_edge(current, next_v)
                current = next_v
                
                if get_degree(current) == 0:
                    break
            
            if len(path) > 1:
                paths.append(path)
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_greedy(edges)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
