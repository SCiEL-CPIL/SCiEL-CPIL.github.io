import random
import time
# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    n = 20
    num_colors = 20
    
    # Edge representation
    # We'll track: for each edge (u,v), its colour, or -1 if absent
    # Use a dict: (u,v) -> colour, with u < v
    edge_color = {}
    
    # For fast 4-cycle checking, track for each pair of vertices the colour
    # and maintain adjacency
    adj = {i: set() for i in range(n)}
    
    def get_color(u, v):
        if u > v:
            u, v = v, u
        return edge_color.get((u, v), -1)
    
    def set_edge(u, v, c):
        if u > v:
            u, v = v, u
        edge_color[(u, v)] = c
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        del edge_color[(u, v)]
        adj[u].discard(v)
        adj[v].discard(u)
        
    def is_valid_with_new_edge(u, v, c):
        """Check if adding edge (u,v) with colour c keeps all 4-cycles rainbow."""
        # A new 4-cycle involving (u,v) must have u-v as one edge.
        # Form: u-v-x-y-u
        # Edges: (u,v), (v,x), (x,y), (y,u)
        
        for x in adj[v]:
            if x == u:
                continue
            cvx = get_color(v, x)
            
            for y in adj[x]:
                if y == u or y == v or y == x:
                    continue
                # Check if (y,u) exists
                if u not in adj[y]:
                    continue
                
                cxy = get_color(x, y)
                cyu = get_color(y, u)
                
                colors = {c, cvx, cxy, cyu}
                if len(colors) != 4:
                    return False
        return True

    def build_graph(seed):
        """Build a graph using a specific random seed for edge ordering."""
        local_edge_color = {}
        local_adj = {i: set() for i in range(n)}
        
        def l_get_color(u, v):
            if u > v:
                u, v = v, u
            return local_edge_color.get((u, v), -1)
            
        def l_set_edge(u, v, c):
            if u > v:
                u, v = v, u
            local_edge_color[(u, v)] = c
            local_adj[u].add(v)
            local_adj[v].add(u)
            
        def l_is_valid(u, v, c):
            for x in local_adj[v]:
                if x == u:
                    continue
                cvx = l_get_color(v, x)
                for y in local_adj[x]:
                    if y == u or y == v or y == x:
                        continue
                    if u not in local_adj[y]:
                        continue
                    cxy = l_get_color(x, y)
                    cyu = l_get_color(y, u)
                    if len({c, cvx, cxy, cyu}) != 4:
                        return False
            return True

        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.seed(seed)
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() > deadline:
                break
            # Try colors in random order
            colors_to_try = list(range(num_colors))
            random.shuffle(colors_to_try)
            for c in colors_to_try:
                if l_is_valid(u, v, c):
                    l_set_edge(u, v, c)
                    break
        
        return local_edge_color, local_adj

    best_edges = 0
    best_result = []
    
    # Try multiple seeds
    for seed in [42, 123, 999, 0, 1, 2]:
        if time.time() > deadline - 5:
            break
        local_edge_color, local_adj = build_graph(seed)
        num_edges = len(local_edge_color)
        if num_edges > best_edges:
            best_edges = num_edges
            best_result = [[u, v, c] for (u, v), c in local_edge_color.items()]
            
    # If no edges found, return empty (should not happen)
    return best_result

# EVOLVE-BLOCK-END
