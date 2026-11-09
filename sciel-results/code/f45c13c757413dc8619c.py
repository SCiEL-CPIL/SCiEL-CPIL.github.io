import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges, seed):
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        used_edges = set()
        paths = []
        all_edges = list(edge_set)
        
        while len(used_edges) < len(all_edges):
            # Find an unused edge
            start_edge = None
            for e in all_edges:
                if e not in used_edges:
                    start_edge = e
                    break
            
            if start_edge is None:
                break # Should not happen
            
            u, v = start_edge
            current_path = [u, v]
            used_edges.add(start_edge)
            
            last = v
            # Extend the path as much as possible while keeping it simple
            while True:
                next_w = None
                for w in adj[last]:
                    e = tuple(sorted((last, w)))
                    if e in used_edges:
                        continue
                    if w in current_path:
                        continue
                    # Found a candidate
                    next_w = w
                    break
                
                if next_w is None:
                    break
                
                current_path.append(next_w)
                used_edges.add(tuple(sorted((last, next_w))))
                last = next_w
                
                # Safety limit if verifier enforces max path length
                if len(current_path) >= 20:
                    break
            
            paths.append(current_path)
        
        return paths

    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        paths = solve_graph(edges, spec[2])
        results.append(paths)
        
    return results
