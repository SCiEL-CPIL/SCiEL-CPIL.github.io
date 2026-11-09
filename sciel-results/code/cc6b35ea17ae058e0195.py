import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Time allocation per n
    time_alloc = {
        46: 60,
        45: 40,
        44: 20,
        43: 10,
        42: 5,
        41: 3,
        40: 2,
        39: 1
    }
    
    for n in range(46, 38, -1):
        min_deg_req = n - 39
        alloc = time_alloc.get(n, 1)
        end_time_for_n = start_time + sum(time_alloc[k] for k in time_alloc if k >= n)
        # Simpler: just track current time
        n_start = time.time()
        
        while time.time() < n_start + alloc:
            # Random restart
            adj = [set() for _ in range(n)]
            edges = []
            
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                ok = True
                for a in adj[u]:
                    if adj[a] & adj[v]:
                        ok = False
                        break
                
                if ok:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            min_deg = min(len(adj[i]) for i in range(n))
            
            if min_deg >= min_deg_req:
                return {'n': n, 'edges': edges}
        
        # If we exhausted time for this n, continue to next smaller n
        # But we must ensure we don't exceed global deadline
        if time.time() > deadline:
            break
    
    return {'n': 39, 'edges': []}
