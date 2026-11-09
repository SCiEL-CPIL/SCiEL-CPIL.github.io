import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    range_n = range(n)
    
    best_edges = []
    best_count = 0
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []
        
        for u, v in all_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            valid = True
            
            # Check triples containing u
            for a in range_n:
                if a == u: continue
                for b in range_n:
                    if b <= a or b == u: continue
                    common = adj[u] & adj[a] & adj[b]
                    if common:
                        c1 = common & (common - 1)
                        if c1:
                            c2 = c1 & (c1 - 1)
                            if c2:
                                valid = False
                                break
                if not valid:
                    break
            
            if valid:
                # Check triples containing v
                for a in range_n:
                    if a == v: continue
                    for b in range_n:
                        if b <= a or b == v: continue
                        common = adj[v] & adj[a] & adj[b]
                        if common:
                            c1 = common & (common - 1)
                            if c1:
                                c2 = c1 & (c1 - 1)
                                if c2:
                                    valid = False
                                    break
                    if not valid:
                        break
            
            if valid:
                current_edges.append([u, v])
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    return best_edges
