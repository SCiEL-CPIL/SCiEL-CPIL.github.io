import time
import random
from collections import defaultdict

def construct():
    random.seed(42) # Ensure deterministic behavior for the heuristic
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_long_cycle(remaining_edges, time_limit=0.5):
        if not remaining_edges:
            return None
        
        adj = defaultdict(list)
        for u, v in remaining_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        vertices = list(set([v for e in remaining_edges for v in e]))
        if not vertices:
            return None
            
        best_cycle = None
        best_len = 0
        start_time = time.time()
        
        attempts = 0
        while time.time() - start_time < time_limit and attempts < 500:
            attempts += 1
            start = random.choice(vertices)
            stack = [(start, [start], None)]
            visited = {start}
            
            while stack:
                node, path, par = stack.pop()
                neighbors = adj[node][:]
                random.shuffle(neighbors)
                
                for nei in neighbors:
                    if nei == par:
                        continue
                    if nei not in visited:
                        visited.add(nei)
                        stack.append((nei, path + [nei], node))
                    else:
                        if nei in path:
                            idx = path.index(nei)
                            cycle_len = len(path) - idx + 1
                            if cycle_len >= 3:
                                if cycle_len > best_len:
                                    best_len = cycle_len
                                    best_cycle = path[idx:]
                                break
                if time.time() - start_time > time_limit:
                    break

        return best_cycle

    def decompose(edges):
        remaining = set(tuple(sorted(e)) for e in edges)
        pieces = []
        start_time = time.time()
        
        while remaining:
            if time.time() - start_time > 140:
                for e in remaining:
                    pieces.append([e[0], e[1]])
                break
            
            cycle = find_long_cycle(remaining, time_limit=0.5)
            if cycle:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        remaining.remove(e)
            else:
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.remove(e)
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
        
    return results
