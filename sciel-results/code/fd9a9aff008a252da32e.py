import random
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead/safety
    
    # Try to find n=30 coloring using backtracking
    result = backtrack(30, start_time, time_limit)
    if result is not None:
        return result
    
    # Fallback to n=29 Paley
    n_base = 29
    q = n_base
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % q in squares:
                colors_base.append(0)
            else:
                colors_base.append(1)
                
    return {'n': n_base, 'colors': colors_base}

def backtrack(n, start_time, time_limit):
    total_edges = n * (n - 1) // 2
    colors = [0] * total_edges
    
    # Adjacency sets for each color
    adj = [set() for _ in range(n)]
    adj1 = [set() for _ in range(n)]
    
    # Precompute edge list
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
            
    def creates_k222_same_part(u, v, c):
        adj_c = adj if c == 0 else adj1
        common = adj_c[u] & adj_c[v]
        common_list = list(common)
        len_common = len(common_list)
        
        for i in range(len_common):
            b1 = common_list[i]
            for j in range(i + 1, len_common):
                b2 = common_list[j]
                candidates = []
                for k in range(len_common):
                    c_val = common_list[k]
                    if c_val == b1 or c_val == b2:
                        continue
                    if c_val in adj_c[b1] and c_val in adj_c[b2]:
                        candidates.append(c_val)
                if len(candidates) >= 2:
                    return True
        return False

    def is_valid_step(u, v, c):
        adj_c = adj if c == 0 else adj1
        # Heuristic: if degree is low, it's safe
        if len(adj_c[u]) < 4 or len(adj_c[v]) < 4:
            return True
        if creates_k222_same_part(u, v, c):
            return False
        return True

    def search(idx):
        if time.time() - start_time > time_limit:
            return False
        if idx == total_edges:
            return True
            
        u, v = edge_list[idx]
        
        # Symmetry breaking: if idx == 0, force color 0
        if idx == 0:
            colors[0] = 0
            adj[u].add(v)
            adj[v].add(u)
            return search(1)
            
        for c in [0, 1]:
            if not is_valid_step(u, v, c):
                continue
                
            colors[idx] = c
            if c == 0:
                adj[u].add(v)
                adj[v].add(u)
            else:
                adj1[u].add(v)
                adj1[v].add(u)
                
            if search(idx + 1):
                return True
                
            if c == 0:
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                adj1[u].remove(v)
                adj1[v].remove(u)
                
        return False

    if search(0):
        return {'n': n, 'colors': colors}
    return None
