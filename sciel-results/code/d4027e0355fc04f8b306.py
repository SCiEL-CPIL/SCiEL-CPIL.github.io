import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    best_edges = set()
    best_count = 0
    
    for seed in range(500):
        if time.time() > deadline:
            break
        random.seed(seed)
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        edge_set = set()
        for u, v in shuffled:
            if time.time() > deadline:
                break
            edge_set.add((u, v))
            if not is_valid(edge_set):
                edge_set.remove((u, v))
        
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = set(edge_set)
    
    return [[u, v] for u, v in sorted(best_edges)]
