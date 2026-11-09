import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=26 with optimized local search
    n = 26
    num_edges = n * (n - 1) // 2  # 325
    
    # Precompute edge endpoints
    edge_u = np.empty(num_edges, dtype=np.int32)
    edge_v = np.empty(num_edges, dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            idx += 1
    
    # Random initialization
    random.seed(42)
    colors = np.random.randint(0, 5, num_edges)
    
    # Build adjacency bitmasks
    adj = [[0] * n for _ in range(5)]
    for e in range(num_edges):
        u = int(edge_u[e])
        v = int(edge_v[e])
        c = int(colors[e])
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
    
    def count_violations():
        count = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                for v in range(u + 1, n):
                    if (mask_u >> v) & 1:
                        common = mask_u & adj[c][v]
                        # Count bits >= 2
                        if common & (common - 1):  # at least 2 bits set
                            count += 1
        return count
    
    initial_violations = count_violations()
    if initial_violations == 0:
        return {'n': n, 'colors': colors.tolist()}
    
    # Local search: iteratively fix violations
    temp = 10.0
    cooling = 0.999
    best_violations = initial_violations
    best_colors = colors.copy()
    
    for iteration in range(2000000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random edge and try to change its color
        e = random.randint(0, num_edges - 1)
        u = int(edge_u[e])
        v = int(edge_v[e])
        old_c = int(colors[e])
        
        # Try each other color
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
        
        # Remove from old color
        adj[old_c][u] &= ~(1 << v)
        adj[old_c][v] &= ~(1 << u)
        
        # Check if new color is valid (no C4 created)
        common = adj[new_c][u] & adj[new_c][v]
        if common & (common - 1):  # 2+ common neighbors
            # Revert
            adj[old_c][u] |= (1 << v)
            adj[old_c][v] |= (1 << u)
            continue
        
        # Accept the move
        colors[e] = new_c
        adj[new_c][u] |= (1 << v)
        adj[new_c][v] |= (1 << u)
        
        # Check if we improved
        if iteration % 1000 == 0:
            current_violations = count_violations()
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                if current_violations == 0:
                    return {'n': n, 'colors': colors.tolist()}
        
        temp *= cooling
    
    # Final check
    final_violations = count_violations()
    if final_violations == 0:
        return {'n': n, 'colors': colors.tolist()}
    
    # Fallback: use best found
    if best_violations == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # Try n=16 construction
    gf4_mult = {(0,0):0,(0,1):0,(0,2):0,(0,3):0,(1,0):0,(1,1):1,(1,2):2,(1,3):3,(2,0):0,(2,1):2,(2,2):3,(2,3):2,(3,0):0,(3,1):3,(3,2):2,(3,3):1}
    gf4_inv = {1:1, 2:3, 3:2}
    def gf4_add(a,b): return a^b
    def gf4_mul(a,b): return gf4_mult[(a,b)]
    def gf4_div(a,b): return gf4_mul(a, gf4_inv[b])
    
    colors_16 = []
    for i in range(16):
        ai, bi = i//4, i%4
        for j in range(i+1, 16):
            aj, bj = j//4, j%4
            d1 = gf4_add(ai, aj)
            d2 = gf4_add(bi, bj)
            if d1 == 0:
                c = 0
            elif d2 == 0:
                c = 1
            else:
                slope = gf4_div(d2, d1)
                c = 1 + slope
            colors_16.append(c)
    
    return {'n': 16, 'colors': colors_16}
