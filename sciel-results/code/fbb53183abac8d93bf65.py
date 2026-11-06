import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _solve(n, time_budget, seed=0):
    rng = random.Random(seed)
    m = n * (n - 1) // 2
    if m == 0:
        return [], 0
    
    # Precompute edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Initialize colors
    colors = [rng.randint(0, 3) for _ in range(m)]
    
    # Build adjacency bitsets
    adj = [[0] * n for _ in range(4)]
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
        
    def _count_triangles():
        count = 0
        for c in range(4):
            for i in range(n):
                neigh_i = adj[c][i] & ~((1 << (i + 1)) - 1)
                while neigh_i:
                    lsb = neigh_i & (-neigh_i)
                    j = lsb.bit_length() - 1
                    neigh_i ^= lsb
                    common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                    count += common.bit_count()
        return count
        
    tri_count = _count_triangles()
    best_tri = tri_count
    best_colors = colors[:]
    
    # Greedy Phase
    greedy_start = time.time()
    improved = True
    while improved and time.time() - greedy_start < time_budget * 0.2 and _time_left() > 2.0:
        improved = False
        for ri in range(m):
            u, v = edges[ri]
            old_c = colors[ri]
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            
            best_new_c = -1
            best_delta = 0
            for new_c in range(4):
                if new_c == old_c:
                    continue
                added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                delta = added - removed
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
                    
            if best_new_c != -1 and best_delta < 0:
                colors[ri] = best_new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[best_new_c][u] |= (1 << v)
                adj[best_new_c][v] |= (1 << u)
                tri_count += best_delta
                improved = True
                
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_colors = colors[:]
                    if best_tri == 0:
                        return best_colors, 0
                        
    # SA Phase
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    sa_start = time.time()
    
    while T > T_min and time.time() - sa_start < time_budget * 0.8 and _time_left() > 2.0:
        ri = rng.randint(0, m - 1)
        u, v = edges[ri]
        old_c = colors[ri]
        new_c = rng.randint(0, 3)
        if new_c == old_c:
            continue
            
        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
        delta = added - removed
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            colors[ri] = new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            tri_count += delta
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = colors[:]
                if best_tri == 0:
                    return best_colors, 0
                    
        T *= cooling
        
    return best_colors, best_tri

def construct():
    # Try n=51
    if _time_left() > 25.0:
        colors, tri = _solve(51, 20.0, seed=123)
        if tri == 0:
            return {'n': 51, 'colors': colors}
            
    # Try n=50
    if _time_left() > 15.0:
        for seed in range(20):
            if _time_left() < 5.0:
                break
            budget = min(5.0, _time_left() - 2.0)
            if budget <= 0:
                break
            colors, tri = _solve(50, budget, seed=seed * 100 + 42)
            if tri == 0:
                return {'n': 50, 'colors': colors}
                
    # Fallback
    if _time_left() > 5.0:
        colors, tri = _solve(49, 3.0, seed=999)
        if tri == 0:
            return {'n': 49, 'colors': colors}
            
    if _time_left() > 2.0:
        colors, tri = _solve(40, 1.0, seed=1)
        if tri == 0:
            return {'n': 40, 'colors': colors}
            
    return {'n': 1, 'colors': [0]}
