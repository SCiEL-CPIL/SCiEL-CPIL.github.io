import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            if i != 255:
                candidates.append(i)
                
    n = len(candidates)
    
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
                
    # Greedy initialization
    def greedy_clique():
        clique = []
        remaining = set(range(n))
        while remaining:
            # Find vertex with max degree in remaining
            best_v = -1
            max_deg = -1
            for v in remaining:
                deg = sum(1 for u in remaining if adj[v][u])
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            if best_v == -1:
                break
            clique.append(best_v)
            remaining = {u for u in remaining if adj[best_v][u]}
        return clique

    best_clique = greedy_clique()
    
    def dfs(clique, candidates_set):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if len(clique) > len(best_clique):
            best_clique = list(clique)
            
        if not candidates_set:
            return
            
        if len(clique) + len(candidates_set) <= len(best_clique):
            return
            
        # Simple ordering: just pick first
        # Better: pick max degree
        v = max(candidates_set, key=lambda u: sum(1 for w in candidates_set if adj[u][w]))
        
        # Include v
        new_candidates = {w for w in candidates_set if adj[v][w]}
        dfs(clique + [v], new_candidates)
        
        # Exclude v
        candidates_set.remove(v)
        dfs(clique, candidates_set)

    dfs([], set(range(n)))
    
    final_set = [255] + [candidates[i] for i in best_clique]
    
    return final_set
