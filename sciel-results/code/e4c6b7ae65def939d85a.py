"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    def to_flat(colors_2d, n):
        result = []
        for i in range(n):
            for j in range(i + 1, n):
                result.append(colors_2d[i][j])
        return result
    
    def check_c6(adj, u, v, c):
        if u == v:
            return False
        au = adj[c][u]
        av = adj[c][v]
        a_mask = au
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            d_mask = av
            while d_mask:
                d_bit = d_mask & (-d_mask)
                d = d_bit.bit_length() - 1
                d_mask ^= d_bit
                if d == a:
                    continue
                common = adj[c][a] & adj[c][d]
                mask_out = (1 << u) | (1 << v) | (1 << a) | (1 << d)
                if common & ~mask_out:
                    return True
        return False
    
    def backtrack_search(n, max_time, seed):
        nonlocal best_n, best_colors
        random.seed(seed)
        colors_2d = [[-1] * n for _ in range(n)]
        adj = [[0] * n for _ in range(4)]
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_assigned = [False] * len(edges)
        
        def assign_edge(idx):
            nonlocal best_n, best_colors
            if time.time() > max_time:
                return False
            if idx == len(edges):
                best_n = n
                best_colors = to_flat(colors_2d, n)
                return True
            
            i, j = edges[idx]
            edge_assigned[idx] = True
            color_order = [0, 1, 2, 3]
            random.shuffle(color_order)
            for c in color_order:
                if check_c6(adj, i, j, c):
                    continue
                colors_2d[i][j] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                if assign_edge(idx + 1):
                    return True
                
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors_2d[i][j] = -1
            edge_assigned[idx] = False
            return False
        
        return assign_edge(0)
    
    # Try to find colorings for increasing n
    for target_n in range(10, 20):
        if time.time() > deadline - 5:
            break
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            if backtrack_search(target_n, deadline - 5, seed=1000 + target_n * 100 + attempt):
                found = True
                break
        if not found:
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
