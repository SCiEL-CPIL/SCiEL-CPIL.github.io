import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    for n in range(47, 41, -1):
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
            
        # Check theoretical bound
        # Max avg degree in C4-free graph is approx (1 + sqrt(4n-3))/2
        max_avg_deg = (1 + math.sqrt(4*n - 3)) / 2
        if target_deg > max_avg_deg:
            continue
            
        # Try multiple restarts
        while time.time() < deadline:
            adj = [set() for _ in range(n)]
            
            # Greedy addition
            while True:
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= target_deg:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                candidates = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if v not in adj[u]:
                            if len(adj[u].intersection(adj[v])) == 0:
                                score = -(len(adj[u]) + len(adj[v]))
                                candidates.append((score, u, v))
                
                if not candidates:
                    break
                    
                # Pick best
                best_score, u, v = min(candidates, key=lambda x: x[0])
                adj[u].add(v)
                adj[v].add(u)
                
    return {'n': 39, 'edges': []}
