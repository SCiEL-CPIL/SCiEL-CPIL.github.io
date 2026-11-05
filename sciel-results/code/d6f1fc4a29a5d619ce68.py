import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges_count = 0
    best_graph = None
    
    while time.time() < deadline:
        random.shuffle(edges)
        
        adj = [0] * n
        edge_count = 0
        
        for u, v in edges:
            safe = True
            
            # Check triples containing u
            # v becomes common neighbor of {u, x, y} if adj[v] has x and y.
            # We need count of common neighbors of u, x, y (excluding v) <= 1.
            # common = adj[u] & adj[x] & adj[y]
            # We need (common & ~(1 << v)).bit_count() <= 1
            
            # Iterate x, y
            # We can iterate over set bits of adj[v]
            # But n is small, just loop.
            
            for x in range(n):
                if x == u or x == v: continue
                if not (adj[v] >> x) & 1: continue
                
                for y in range(x + 1, n):
                    if y == u or y == v: continue
                    if not (adj[v] >> y) & 1: continue
                    
                    common = adj[u] & adj[x] & adj[y]
                    # Exclude v
                    common &= ~(1 << v)
                    if common.bit_count() > 1:
                        safe = False
                        break
                if not safe: break
            
            if not safe:
                # Check triples containing v
                for x in range(n):
                    if x == u or x == v: continue
                    if not (adj[u] >> x) & 1: continue
                    
                    for y in range(x + 1, n):
                        if y == u or y == v: continue
                        if not (adj[u] >> y) & 1: continue
                        
                        common = adj[v] & adj[x] & adj[y]
                        common &= ~(1 << u)
                        if common.bit_count() > 1:
                            safe = False
                            break
                    if not safe: break
            
            if safe:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            # Reconstruct edges
            best_graph = []
            for i in range(n):
                mask = adj[i]
                for j in range(i + 1, n):
                    if (mask >> j) & 1:
                        best_graph.append([i, j])
            
    return best_graph if best_graph else []
