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
        
        # Build initial adjacency and degree counts
        remaining_edges = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        
        # Helper to get current degree
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
            
            # Find a starting vertex. 
            # Strategy: Pick a vertex with degree > 0. 
            # To potentially get longer paths, we might pick a vertex with high degree, 
            # or just the first one found. 
            # Let's try to pick a vertex that is an endpoint of a "bridge" or just arbitrary.
            # Simple: Pick the first vertex in the set of vertices with degree > 0.
            start_v = None
            for v in adj:
                if get_degree(v) > 0:
                    start_v = v
                    break
                    
            if start_v is None:
                break
                
            path = [start_v]
            current = start_v
            
            while True:
                # Find next vertex
                neighbors = list(adj[current])
                if not neighbors:
                    break
                    
                # Heuristic: Choose neighbor with the LOWEST degree.
                # Rationale: Clearing leaves early prevents them from being stranded.
                # This is a common heuristic for path/edge cover problems.
                next_v = min(neighbors, key=lambda x: get_degree(x))
                
                # Add to path
                path.append(next_v)
                remove_edge(current, next_v)
                current = next_v
                
                # If current has no more edges, the path ends
                if get_degree(current) == 0:
                    break
            
            if len(path) > 1:
                paths.append(path)
            else:
                # Should not happen if remaining_edges is not empty and start_v had degree > 0
                # But just in case, handle isolated edge case
                pass
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_greedy(edges)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
