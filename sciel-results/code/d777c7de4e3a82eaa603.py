import time
import random
from collections import defaultdict

def instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges, n, num_restarts=20, time_limit=100.0):
    start_time = time.time()
    edge_set = set(tuple(sorted(e)) for e in edges)
    best_decomposition = None
    best_count = float('inf')
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        remaining_edges = set(edge_set)
        paths = []
        
        while remaining_edges:
            sub_adj = defaultdict(set)
            nodes_in_subgraph = set()
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
                nodes_in_subgraph.add(u)
                nodes_in_subgraph.add(v)
            
            best_path = []
            
            def dfs_random(start_node):
                nonlocal best_path
                path = [start_node]
                visited = {start_node}
                
                def _dfs(node):
                    nonlocal best_path
                    if len(path) > len(best_path):
                        best_path = path[:]
                    neighbors = list(sub_adj[node])
                    random.shuffle(neighbors)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            _dfs(neighbor)
                            path.pop()
                            visited.remove(neighbor)
                            
                _dfs(start_node)
                
            for start in nodes_in_subgraph:
                dfs_random(start)
                
            if not best_path:
                # Fallback if no path found (should not happen)
                break
                
            paths.append(best_path)
            
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i+1]
                edge = tuple(sorted((u, v)))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    
        if len(paths) < best_count:
            best_count = len(paths)
            best_decomposition = paths
            
    if best_decomposition is None:
        # Return trivial decomposition
        return [[list(e) for e in edge_set]]
        
    return best_decomposition

def construct():
    start_time = time.time()
    deadline = 110.0
    
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            # Fallback
            return [[list(e) for e in instance_graph(*spec)] for spec in specs]
            
        edges = instance_graph(n, p, seed)
        
        # Allocate time proportionally to difficulty (N^2 or edges)
        # N=30 is hardest.
        if n <= 25:
            restarts = 50
            time_limit = 30.0
        else:
            restarts = 20
            time_limit = 50.0
            
        decomposition = decompose_graph(edges, n, restarts, time_limit)
        results.append(decomposition)
        
    return results
