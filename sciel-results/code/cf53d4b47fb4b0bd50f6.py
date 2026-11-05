import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def is_c4_free(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return False
        return True
    
    def check_c4_conflict(adj, u, v):
        for a in adj[v]:
            if a == u:
                continue
            for b in adj[u]:
                if b == a or b == v:
                    continue
                if b in adj[a]:
                    return True
        return False
    
    def greedy_build(n, seed, min_deg_needed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edge_set = set()
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() > deadline:
                return None
            if not check_c4_conflict(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= min_deg_needed:
            edges = [[u, v] for (u, v) in edge_set]
            return {'n': n, 'edges': edges}
        return None
    
    # Try n=47
    for seed in range(200):
        if time.time() > deadline - 15:
            break
        result = greedy_build(47, seed, 8)
        if result is not None:
            return result
    
    # Try n=46
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        result = greedy_build(46, seed, 7)
        if result is not None:
            return result
    
    # Try n=45
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        result = greedy_build(45, seed, 6)
        if result is not None:
            return result
    
    # Fallback
    return {'n': 39, 'edges': []}
