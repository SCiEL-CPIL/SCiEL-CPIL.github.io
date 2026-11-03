import random
import time
from collections import deque

def construct(k, g):
    # ... (Moore bound calc) ...
    
    best_n = float('inf')
    best_edges = None
    
    start_time = time.time()
    time_limit = 90.0 # Leave 30s for fallback if needed
    
    seed = 0
    while time.time() - start_time < time_limit:
        random.seed(seed)
        seed += 1
        
        adj = []
        n = 0
        edges = []
        
        def add_vertex():
            nonlocal n
            adj.append(set())
            n += 1
            return n - 1
            
        def add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            
        def dist(u, v, max_dist):
            if u == v: return 0
            visited = [-1] * n
            visited[u] = 0
            queue = deque([(u, 0)])
            while queue:
                node, d = queue.popleft()
                if d > max_dist: continue
                for nei in adj[node]:
                    if visited[nei] == -1:
                        if nei == v:
                            return d + 1
                        visited[nei] = d + 1
                        queue.append((nei, d + 1))
            return -1
            
        v0 = add_vertex()
        
        max_iter = 10000
        for _ in range(max_iter):
            candidates = [i for i in range(n) if len(adj[i]) < k]
            if not candidates: break
            
            v = random.choice(candidates)
            
            found_u = -1
            # Strategy: If n is small, check all. Else, check random sample.
            if n < 50:
                for u in range(n):
                    if u == v or u in adj[v]: continue
                    if dist(v, u, g-2) == -1:
                        found_u = u
                        break
            else:
                # Sample 20 random vertices
                for _ in range(min(20, n-1)):
                    u = random.randint(0, n-1)
                    if u == v or u in adj[v]: continue
                    if dist(v, u, g-2) == -1:
                        found_u = u
                        break
            
            if found_u != -1:
                add_edge(v, found_u)
            else:
                w = add_vertex()
                add_edge(v, w)
                
        if len(candidates) == 0:
            # Valid graph
            if n < best_n:
                best_n = n
                best_edges = list(edges)
                
    if best_n != float('inf'):
        return best_n, best_edges
    else:
        return _luw_fallback(k, g)

# Include LUW fallback code
