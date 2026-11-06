import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges_total = len(all_edges)
    
    best_edges = []
    best_count = 0
    
    # Run multiple greedy constructions
    while time.time() - start_time < time_limit:
        random.shuffle(all_edges)
        
        # adj[u][v] = color, -1 if no edge
        adj = [[-1] * n for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            forbidden = set()
            
            # Check all paths u-w-x-v of length 3
            # w, x are distinct and not u, v
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w] == -1:
                    continue
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if adj[w][x] == -1:
                        continue
                    if adj[x][v] == -1:
                        continue
                    
                    # Path u-w-x-v exists.
                    # Adding (u,v) creates cycle u-v-x-w-u.
                    # Edges: (u,v), (v,x), (x,w), (w,u).
                    # Colors of existing edges in this cycle:
                    forbidden.add(adj[v][x])
                    forbidden.add(adj[x][w])
                    forbidden.add(adj[w][u])
            
            # Find a valid color
            # We can pick randomly from valid colors to explore different solutions
            valid_colors = [c for c in range(num_colors) if c not in forbidden]
            
            if valid_colors:
                c = random.choice(valid_colors)
                adj[u][v] = c
                adj[v][u] = c
                current_edges.append([u, v, c])
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    return best_edges
