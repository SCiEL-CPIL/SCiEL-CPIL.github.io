import time
import random

def construct():
    n = 20
    num_colors = 20
    # Use a safety margin to ensure we return before the 150s limit
    deadline = time.time() + 145
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # Helper to check if adding (u, v, c) is valid
    # adj is a list of dicts: adj[u][v] = color
    def is_valid_addition(adj, u, v, c):
        # Check all 4-cycles containing edge (u, v)
        # Cycle: u - v - x - y - u
        # Edges: (u,v), (v,x), (x,y), (y,u)
        
        # Iterate neighbors of v
        for x, c_vx in adj[v].items():
            if x == u:
                continue
            # Iterate neighbors of u
            for y, c_yu in adj[u].items():
                if y == v or y == x:
                    continue
                # Check if (x, y) is an edge
                if y in adj[x]:
                    c_xy = adj[x][y]
                    # Colors: c (for u,v), c_vx, c_xy, c_yu
                    # They must be distinct
                    if c == c_vx or c == c_xy or c == c_yu or c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                        return False
        return True

    # Greedy construction with random restarts
    while time.time() < deadline:
        random.shuffle(all_edges)
        # adj[u] is a dict of neighbors and colors
        adj = [{} for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            # Try to find a valid color
            # We can try colors in random order to diversify
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            added = False
            for c in colors:
                if is_valid_addition(adj, u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    current_edges.append([u, v, c])
                    added = True
                    break
            
            # If no color works, we skip this edge in this restart
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
    
    return best_edges
