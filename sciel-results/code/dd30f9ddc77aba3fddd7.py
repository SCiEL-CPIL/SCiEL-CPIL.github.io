import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    r = g // 2
    if g % 2 == 0:
        n_min = 2 * sum((k - 1) ** i for i in range(r))
    else:
        n_min = (k - 1) * sum((k - 1) ** i for i in range(r))
        
    if n_min % 2 != 0:
        n_min += 1
        
    max_n = 1100
    
    for n in range(n_min, max_n + 1, 2):
        if time.time() > deadline:
            break
            
        n_start_time = time.time()
        
        # Try a few attempts
        for attempt in range(5):
            if time.time() - n_start_time > 0.5:
                break
                
            adj = [set() for _ in range(n)]
            deg = [0] * n
            edges_added = 0
            target_edges = (n * k) // 2
            
            mark = [0] * n
            iter_id = 0
            
            def has_short_path_fast(u, v, max_depth):
                nonlocal iter_id
                iter_id += 1
                if u == v:
                    return True
                mark[u] = iter_id
                queue = deque([(u, 0)])
                while queue:
                    node, dist = queue.popleft()
                    if dist == max_depth:
                        continue
                    for neighbor in adj[node]:
                        if neighbor == v:
                            return True
                        if mark[neighbor] != iter_id:
                            mark[neighbor] = iter_id
                            queue.append((neighbor, dist + 1))
                return False

            while edges_added < target_edges:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                
                if u == v or deg[u] >= k or deg[v] >= k or v in adj[u]:
                    continue
                
                if has_short_path_fast(u, v, g - 2):
                    continue
                
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges_added += 1
                
            if any(d != k for d in deg):
                continue
                
            edge_list = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edge_list.append((i, j))
            
            return n, edge_list

    return n_min, []
