import random
import time

def construct():
    n = 20
    colors = 20
    deadline = time.time() + 120 # Safety margin
    
    best_edges = 0
    best_result = []
    
    # Pre-generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        # adjacency matrix: adj[u][v] = color or -1
        adj = [[-1]*n for _ in range(n)]
        edge_count = 0
        valid = True
        
        for u, v in all_edges:
            # Check if we can add (u, v)
            forbidden = set()
            can_add = True
            
            # Check all 4-cycles completed by (u, v)
            # Cycle: u - v - w - x - u
            for w in range(n):
                if w == u or w == v: continue
                if adj[v][w] == -1: continue
                c_vw = adj[v][w]
                
                for x in range(n):
                    if x == u or x == v or x == w: continue
                    if adj[w][x] == -1: continue
                    if adj[x][u] == -1: continue
                    
                    c_wx = adj[w][x]
                    c_xu = adj[x][u]
                    
                    # The three existing edges must have distinct colours
                    if c_vw == c_wx or c_wx == c_xu or c_vw == c_xu:
                        can_add = False
                        break
                    
                    # The new edge cannot have these colours
                    forbidden.add(c_vw)
                    forbidden.add(c_wx)
                    forbidden.add(c_xu)
            
            if not can_add:
                continue
                
            if len(forbidden) >= colors:
                continue
                
            # Pick a random valid colour
            available = [c for c in range(colors) if c not in forbidden]
            random.shuffle(available)
            c_new = available[0]
            
            # Add edge
            adj[u][v] = c_new
            adj[v][u] = c_new
            edge_count += 1
            
        if edge_count > best_edges:
            best_edges = edge_count
            # Reconstruct result
            result = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] != -1:
                        result.append([i, j, adj[i][j]])
            best_result = result
            
            if best_edges > 150: # Heuristic: if very dense, stop
                break

    return best_result
