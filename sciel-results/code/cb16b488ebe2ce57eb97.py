import random
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        if not edges:
            results.append([])
            continue
            
        # Build adjacency list
        # We need a mutable structure for the greedy algorithm
        # adj[u] = set of v such that (u,v) in remaining_edges
        # remaining_edges = set of (u,v) with u<v
        
        best_paths = None
        best_count = float('inf')
        
        # Number of restarts depends on remaining time
        # Estimate: each restart is fast. Let's do up to 5000 restarts or until time is low.
        
        for restart in range(5000):
            if time.time() > deadline - 5.0:
                break
                
            # Deep copy of edges for this run
            rem_edges = set(edges)
            adj = {i: set() for i in range(n)}
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            paths = []
            
            # Greedy path finding
            while rem_edges:
                # Find start vertex
                # Heuristic: pick vertex with max degree in current graph
                # If tie, random
                max_deg = -1
                candidates = []
                for v in range(n):
                    deg = len(adj[v])
                    if deg > 0:
                        if deg > max_deg:
                            max_deg = deg
                            candidates = [v]
                        elif deg == max_deg:
                            candidates.append(v)
                
                if not candidates:
                    break # Should not happen if rem_edges not empty
                
                start = random.choice(candidates)
                path = [start]
                current = start
                
                while True:
                    neighbors = adj[current]
                    if not neighbors:
                        break
                    
                    # Heuristic: pick neighbor with min degree (Warnsdorff)
                    # This tends to keep options open for the rest of the graph
                    min_deg = float('inf')
                    next_candidates = []
                    for nb in neighbors:
                        deg = len(adj[nb])
                        if deg < min_deg:
                            min_deg = deg
                            next_candidates = [nb]
                        elif deg == min_deg:
                            next_candidates.append(nb)
                    
                    next_v = random.choice(next_candidates)
                    path.append(next_v)
                    
                    # Remove edge (current, next_v)
                    if current < next_v:
                        rem_edges.remove((current, next_v))
                    else:
                        rem_edges.remove((next_v, current))
                    
                    adj[current].remove(next_v)
                    adj[next_v].remove(current)
                    
                    current = next_v
                
                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
                
                # Early stop if we hit the theoretical lower bound?
                # Lower bound is max(1, odd/2).
                # Let's calculate odd count once.
        
        # Calculate theoretical lower bound for early stopping or just info
        degs = [0]*n
        for u,v in edges:
            degs[u]+=1
            degs[v]+=1
        odd_count = sum(1 for d in degs if d % 2 != 0)
        lower_bound = max(1, odd_count // 2)
        
        if best_paths is None:
             # Fallback: single edge paths
             best_paths = [[u,v] for u,v in edges]
        
        results.append(best_paths)
        
    return results

from random import Random
