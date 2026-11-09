import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def has_path_len4(adj_mask, u, w, limit):
        curr = {u}
        visited = {u}
        
        for _ in range(4):
            next_set = set()
            for node in curr:
                mask = adj_mask[node]
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v not in visited:
                        visited.add(v)
                        next_set.add(v)
                    m ^= lsb
            curr = next_set
            if not curr:
                return False
        
        return w in curr

    def compute_p4(adj_masks, limit):
        p4 = [set() for _ in range(4)]
        for u in range(limit):
            for w in range(u + 1, limit):
                for c in range(4):
                    if has_path_len4(adj_masks[c], u, w, limit):
                        p4[c].add((u, w))
        return p4

    def solve_n(n, time_limit):
        end_time = time.time() + time_limit
        m = n * (n - 1) // 2
        
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def get_edge_idx(i, j):
            if i > j: i, j = j, i
            return edge_map[(i, j)]
        
        def backtrack(v):
            if time.time() > end_time:
                return False
            
            if v == n:
                return True
            
            p4 = compute_p4(adj_masks, v)
            
            colors_for_v = [None] * v
            
            def solve_u(u):
                if time.time() > end_time:
                    return False
                
                if u == v:
                    return True
                
                colors_to_try = [0, 1, 2, 3]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    valid = True
                    for w in range(u):
                        if colors_for_v[w] == c:
                            if (u, w) in p4[c] or (w, u) in p4[c]:
                                valid = False
                                break
                    
                    if valid:
                        colors_for_v[u] = c
                        edge_idx = get_edge_idx(v, u)
                        colors[edge_idx] = c
                        adj_masks[c][v] |= (1 << u)
                        adj_masks[c][u] |= (1 << v)
                        
                        if solve_u(u + 1):
                            return True
                        
                        colors[edge_idx] = -1
                        adj_masks[c][v] &= ~(1 << u)
                        adj_masks[c][u] &= ~(1 << v)
                        
                return False
            
            return solve_u(0)
        
        colors = [-1] * m
        adj_masks = [[0] * n for _ in range(4)]
        
        max_restarts = 50
        for _ in range(max_restarts):
            if time.time() > end_time:
                break
            
            colors = [-1] * m
            adj_masks = [[0] * n for _ in range(4)]
            
            if backtrack(0):
                return colors
        
        return None

    target_ns = [17, 18, 19, 20, 16, 15, 14, 13, 12, 11, 10]
    
    best_n = 9
    best_colors = None
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
            
        if n >= 17:
            t_lim = 10
        elif n >= 15:
            t_lim = 5
        else:
            t_lim = 2
            
        res = solve_n(n, t_lim)
        if res:
            best_n = n
            best_colors = res
            
    if best_colors is None:
        n = 5
        m = n * (n - 1) // 2
        best_colors = [random.randint(0, 3) for _ in range(m)]
        best_n = n
        
    return {'n': best_n, 'colors': best_colors}
