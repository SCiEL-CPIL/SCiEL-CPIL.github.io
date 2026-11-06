"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math

def _mul(a, b):
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_k32_colors():
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return colors

def _sa_find_coloring(n, deadline, seed=42):
    """
    Simulated Annealing to find a 4-coloring of K_n with no monochromatic triangles.
    Returns (n, colors) if found, else None.
    """
    rng = random.Random(seed)
    
    # Initialize matrix
    mat = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = rng.randint(0, 3)
            mat[i][j] = c
            mat[j][i] = c
            
    # Pre-compute list of neighbors for each vertex? 
    # Not strictly necessary, but helps.
    
    def get_delta(u, v, new_c):
        old_c = mat[u][v]
        if old_c == new_c:
            return 0
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            cuw = mat[u][w]
            cvw = mat[v][w]
            if old_c == cuw and old_c == cvw:
                delta -= 1
            if new_c == cuw and new_c == cvw:
                delta += 1
        return delta

    # Initial count
    total_mono = 0
    for i in range(n):
        for j in range(i+1, n):
            c = mat[i][j]
            for k in range(j+1, n):
                if mat[i][k] == c and mat[j][k] == c:
                    total_mono += 1
                    
    best_total = total_mono
    best_mat = [row[:] for row in mat]
    
    T = 5.0
    cooling = 0.9995
    
    start_time = time.time()
    
    while time.time() < deadline:
        u = rng.randint(0, n-1)
        v = rng.randint(0, n-1)
        if u == v:
            continue
        
        new_c = rng.randint(0, 3)
        
        delta = get_delta(u, v, new_c)
        
        if delta <= 0:
            mat[u][v] = new_c
            mat[v][u] = new_c
            total_mono += delta
            if total_mono < best_total:
                best_total = total_mono
                best_mat = [row[:] for row in mat]
                if best_total == 0:
                    break
        else:
            # Avoid overflow in exp
            if delta / T < 700: 
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                mat[u][v] = new_c
                mat[v][u] = new_c
                total_mono += delta
                if total_mono < best_total:
                    best_total = total_mono
                    best_mat = [row[:] for row in mat]
                    if best_total == 0:
                        break
        T *= cooling
        
    if best_total == 0:
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(best_mat[i][j])
        return n, colors
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find 50-vertex coloring
    # We allocate time for multiple restarts
    # If we find one, we return immediately.
    
    # Try 50 vertices
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        res = _sa_find_coloring(50, deadline - 5, seed)
        if res:
            return {'n': 50, 'colors': res[1]}
            
    # If 50 fails, try 49?
    # The score is value/seed. 49/32 = 1.53. 32/32 = 1.0.
    # So 49 is better than 32.
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        res = _sa_find_coloring(49, deadline - 2, seed)
        if res:
            return {'n': 49, 'colors': res[1]}
            
    # Fallback to 32
    base_colors = _build_k32_colors()
    return {'n': 32, 'colors': base_colors}

# EVOLVE-BLOCK-END
