import random
import time

def construct():
    n = 20
    num_colors = 20
    deadline = time.time() + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # To speed up neighbor lookups, we can maintain a list of neighbors
    # But n=20 is small, so direct adjacency matrix check is fine.
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj = [[-1] * n for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            forbidden = set()
            
            # Check all x, y
            # x is a neighbor of u
            # y is a neighbor of v
            # (x, y) must be an edge
            # Cycle: u-v-y-x-u
            
            # Iterate x
            for x in range(n):
                if x == u or x == v:
                    continue
                if adj[u][x] == -1:
                    continue
                
                # Iterate y
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    if adj[y][v] == -1:
                        continue
                    if adj[x][y] == -1:
                        continue
                    
                    # Found path u-x-y-v
                    # Colors to avoid: c(v,y), c(y,x), c(x,u)
                    forbidden.add(adj[y][v])
                    forbidden.add(adj[x][y])
                    forbidden.add(adj[u][x])
                    
                    if len(forbidden) == num_colors:
                        break
                if len(forbidden) == num_colors:
                    break
            
            if len(forbidden) < num_colors:
                # Pick a valid color
                # Optimization: precompute valid colors list?
                # Just filter is fine for 20 items.
                valid_colors = [c for c in range(num_colors) if c not in forbidden]
                c = random.choice(valid_colors)
                adj[u][v] = c
                adj[v][u] = c
                current_edges.append([u, v, c])
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges

    return best_edges
