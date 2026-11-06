import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = []
    best_count = 0
    
    def can_add_edge(adj_mask, color_mat, u, v):
        """Check if edge (u,v) can be added. Returns (valid, forbidden_mask)."""
        forbidden = 0
        mask_v = adj_mask[v]
        mask_u = adj_mask[u]
        
        for w in range(n):
            if w == u: continue
            if not ((mask_v >> w) & 1): continue
            
            c_vw = color_mat[v][w]
            
            for x in range(n):
                if x == v: continue
                if x == w: continue
                if not ((mask_u >> x) & 1): continue
                if not ((adj_mask[w] >> x) & 1): continue
                
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return False, 0
                
                forbidden |= (1 << c_vw) | (1 << c_wx) | (1 << c_xu)
        
        return True, forbidden
    
    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        adj_mask = [0] * n
        color_mat = [[-1]*n for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            valid, forbidden = can_add_edge(adj_mask, color_mat, u, v)
            if not valid:
                continue
            
            # Find a valid color
            valid_colors = [c for c in range(num_colors) if not (forbidden >> c) & 1]
            if valid_colors:
                c = random.choice(valid_colors)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                color_mat[u][v] = c
                color_mat[v][u] = c
                edges.append([u, v, c])
        
        return edges
    
    def local_search(edges, seed=None):
        """Try to add missing edges to an existing solution."""
        if seed is not None:
            random.seed(seed)
        
        adj_mask = [0] * n
        color_mat = [[-1]*n for _ in range(n)]
        for u, v, c in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            color_mat[u][v] = c
            color_mat[v][u] = c
        
        # Build list of missing edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_mask[i] >> j) & 1)]
        random.shuffle(missing)
        
        improved = True
        while improved:
            improved = False
            for u, v in missing[:]:
                valid, forbidden = can_add_edge(adj_mask, color_mat, u, v)
                if not valid:
                    continue
                
                valid_colors = [c for c in range(num_colors) if not (forbidden >> c) & 1]
                if valid_colors:
                    c = random.choice(valid_colors)
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    color_mat[u][v] = c
                    color_mat[v][u] = c
                    edges.append([u, v, c])
                    missing.remove((u, v))
                    improved = True
        
        return edges

    restarts = 0
    while time.time() < deadline:
        edges = run_greedy(seed=restarts)
        if time.time() < deadline - 5:
            edges = local_search(edges, seed=restarts * 1000 + 42)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        if best_count == 190:
            break
            
        restarts += 1

    return best_edges
