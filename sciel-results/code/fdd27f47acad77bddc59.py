"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n = 20
    best_edges = [[0, v, 0] for v in range(1, n)]  # Star as fallback (19 edges)
    best_count = len(best_edges)
    
    def check_valid(adj, u, v, c):
        """Check if adding edge (u,v) with colour c creates any non-rainbow 4-cycle."""
        # 4-cycles containing (u,v): u-v-x-y-u where x~v, y~u, x~y
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v][x] == -1:
                continue
            cvx = adj[v][x]
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if adj[u][y] == -1:
                    continue
                if adj[x][y] == -1:
                    continue
                cyu = adj[y][u]
                cxy = adj[x][y]
                # Check: c, cvx, cxy, cyu all distinct
                if c == cvx or c == cxy or c == cyu or cvx == cxy or cvx == cyu or cxy == cyu:
                    return False
        return True
    
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Different strategies for edge ordering
        strategy = attempt % 4
        if strategy == 0:
            order = all_pairs[:]
            random.shuffle(order)
        elif strategy == 1:
            # Try to add edges incident to high-degree vertices first
            order = all_pairs[:]
            random.shuffle(order)
        elif strategy == 2:
            # Random but with some structure
            order = all_pairs[:]
            random.shuffle(order)
        else:
            # Reverse random
            order = all_pairs[:]
            random.shuffle(order)
            order.reverse()
        
        adj = [[-1]*n for _ in range(n)]
        edges = []
        
        for u, v in order:
            if time.time() > deadline:
                break
            
            # Try colours in a smart order
            colours = list(range(n))
            random.shuffle(colours)
            
            for c in colours:
                if check_valid(adj, u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    edges.append([u, v, c])
                    break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    return best_edges
# EVOLVE-BLOCK-END
