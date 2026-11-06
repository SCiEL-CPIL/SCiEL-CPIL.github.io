import random
import time

def construct():
    n = 20
    deadline = time.time() + 110
    best_edges = 0
    best_adj = None
    
    # Precompute all edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        random.shuffle(edges)
        adj = [0] * n
        current_edges = 0
        
        for u, v in edges:
            # Check if u, v already connected
            if (adj[u] >> v) & 1:
                continue
                
            # Check validity
            valid = True
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(x+1, n):
                    if y == u or y == v:
                        continue
                    # Check triple (u, x, y)
                    # v becomes common neighbor if v in N(x) and v in N(y)
                    if ((adj[x] >> v) & 1) and ((adj[y] >> v) & 1):
                        common = adj[u] & adj[x] & adj[y]
                        if common.bit_count() >= 2:
                            valid = False
                            break
                    if not valid:
                        break
                    # Check triple (v, x, y)
                    # u becomes common neighbor if u in N(x) and u in N(y)
                    if ((adj[x] >> u) & 1) and ((adj[y] >> u) & 1):
                        common = adj[v] & adj[x] & adj[y]
                        if common.bit_count() >= 2:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
                
    return result
