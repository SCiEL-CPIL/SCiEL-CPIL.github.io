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
        # A 4-cycle containing (u,v) is u-v-x-y-u.
        # x must be neighbor of v, y must be neighbor of u, and x-y must be an edge.
        # Colors c(u,v), c(v,x), c(x,y), c(y,u) must be distinct.
        
        # Pre-fetch neighbors for speed
        # We iterate x over neighbors of v (excluding u)
        # We iterate y over neighbors of u (excluding u, v, x)
        # Check if x-y is an edge.
        
        for x in range(n):
            if x == u or x == v:
                continue
            cvx = adj[v][x]
            if cvx == -1:
                continue
            
            # Check color conflict with (u,v) and (v,x)
            if c == cvx:
                return False
            
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                
                cyu = adj[u][y]
                if cyu == -1:
                    continue
                
                # Check color conflicts among (u,v), (v,x), (y,u)
                # c, cvx, cyu must be distinct
                if c == cyu or cvx == cyu:
                    return False
                
                # Check if x and y are connected
                cxy = adj[x][y]
                if cxy == -1:
                    continue
                
                # Check color conflict with (x,y)
                # c, cvx, cyu, cxy must be distinct
                if c == cxy or cvx == cxy or cyu == cxy:
                    return False
        
        return True
    
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Run multiple attempts with different random seeds/orders to find a dense valid graph
    # The check is O(N^2) per edge, which is fast enough for N=20.
    max_attempts = 5000 # High number, but time-bounded
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        adj = [[-1]*n for _ in range(n)]
        edges = []
        
        # Use a random order for edges
        order = all_pairs[:]
        random.shuffle(order)
        
        for u, v in order:
            if time.time() > deadline:
                break
            
            # Try colours in a random order
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
            
            # If we reach a very high density, we might stop early if time permits?
            # Or just let it run until deadline.
    
    return best_edges
# EVOLVE-BLOCK-END
