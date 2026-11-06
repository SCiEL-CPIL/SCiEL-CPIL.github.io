import time
import numpy as np
import random

def construct():
    deadline = time.time() + 115
    n = 50
    # 4 colors: 0, 1, 2, 3
    # Edges are (i, j) with i < j
    # Total edges: 50 * 49 // 2 = 1225
    
    # Initialize random coloring
    colors = np.random.randint(0, 4, size=(n, n), dtype=np.int8)
    # Symmetrize and zero diagonal
    colors = np.triu(colors, k=1)
    colors = colors + colors.T
    np.fill_diagonal(colors, -1)
    
    # Function to count mono triangles
    def count_mono_triangles(colors):
        # For each color c, count triangles in the subgraph of color c
        total = 0
        for c in range(4):
            # Mask where color is c
            mask = (colors == c)
            # Triangle count: sum over i<j<k of mask[i,j]*mask[j,k]*mask[i,k]
            # Using matrix multiplication:
            # A = mask
            # Triangles = trace(A^3) / 6 ? No, for undirected simple graph.
            # Number of triangles in graph with adjacency matrix A is trace(A^3)/6.
            # But A is 0/1.
            # This is O(n^3) per color, 4 * 50^3 = 500k ops. Very fast.
            # However, we can use a faster bitset method or just numpy.
            A = mask.astype(np.int32)
            # A^3 diagonal sum
            # A2 = A @ A
            # trace(A @ A2)
            # This is fast in numpy.
            A2 = A @ A
            trace_val = np.trace(A @ A2)
            total += trace_val // 6
        return total

    best_colors = colors.copy()
    best_count = count_mono_triangles(best_colors)
    
    if best_count == 0:
        # Extract upper triangle
        res_colors = []
        for i in range(n):
            for j in range(i+1, n):
                res_colors.append(int(colors[i, j]))
        return {'n': n, 'colors': res_colors}

    # Local Search
    rng = random.Random(42)
    current_colors = best_colors.copy()
    current_count = best_count
    
    start_time = time.time()
    step = 0
    while time.time() < deadline - 5:
        step += 1
        # Pick a random edge to flip
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_c = current_colors[i, j]
        new_c = rng.randint(0, 3)
        if old_c == new_c:
            continue
            
        # Calculate change in mono triangles
        # Only triangles involving edge (i, j) can change.
        # Triangles (i, j, k) for all k != i, j.
        # There are n-2 such triangles.
        # For each k, the triangle (i, j, k) is mono if:
        # 1. colors[i,k] == colors[j,k] == old_c (and we change (i,j) to new_c) -> becomes non-mono if new_c != old_c
        # 2. colors[i,k] == colors[j,k] == new_c (and we change (i,j) to new_c) -> becomes mono
        # 3. Other cases don't affect mono-ness of this specific triangle.
        
        delta = 0
        # We need to check all k
        # Vectorized check:
        # col_i = current_colors[i, :]
        # col_j = current_colors[j, :]
        # We want to count k where col_i[k] == col_j[k] == old_c
        # and k where col_i[k] == col_j[k] == new_c
        
        # Mask for k != i, j
        mask_k = np.ones(n, dtype=bool)
        mask_k[i] = False
        mask_k[j] = False
        
        # Count triangles that were mono with old_c and will not be
        # Triangle (i, j, k) is mono old_c if:
        # current_colors[i, k] == old_c AND current_colors[j, k] == old_c
        # AND current_colors[i, j] == old_c (which is true)
        
        # After flip, current_colors[i, j] == new_c.
        # Triangle is mono new_c if:
        # current_colors[i, k] == new_c AND current_colors[j, k] == new_c
        
        # So, we lose triangles where neighbors match old_c.
        # We gain triangles where neighbors match new_c.
        
        # Note: If old_c == new_c, no change. We already skipped that.
        
        # Count neighbors matching old_c
        match_old = (current_colors[i, mask_k] == old_c) & (current_colors[j, mask_k] == old_c)
        lost = np.sum(match_old)
        
        # Count neighbors matching new_c
        match_new = (current_colors[i, mask_k] == new_c) & (current_colors[j, mask_k] == new_c)
        gained = np.sum(match_new)
        
        delta = gained - lost
        
        # Accept if delta < 0 (improvement) or with some probability (annealing)
        # Simple hill climbing first
        if delta < 0:
            current_colors[i, j] = new_c
            current_colors[j, i] = new_c
            current_count += delta
            
            if current_count < best_count:
                best_count = current_count
                best_colors = current_colors.copy()
                
            if best_count == 0:
                res_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        res_colors.append(int(best_colors[i, j]))
                return {'n': n, 'colors': res_colors}
        
        # If stuck for a while, kick
        if step % 10000 == 0:
            # Randomly flip 10 edges
            for _ in range(10):
                ii = rng.randint(0, n - 1)
                jj = rng.randint(0, n - 1)
                if ii != jj:
                    if ii > jj:
                        ii, jj = jj, ii
                    current_colors[ii, jj] = rng.randint(0, 3)
                    current_colors[jj, ii] = current_colors[ii, jj]
            current_count = count_mono_triangles(current_colors)
            if current_count < best_count:
                best_count = current_count
                best_colors = current_colors.copy()

    # If we didn't find 50, fallback to 32
    # Reconstruct K32
    def _mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
        
    n32 = 32
    colors32 = []
    for i in range(n32):
        for j in range(i + 1, n32):
            if i // 16 == j // 16:
                colors32.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors32.append(3)
                
    return {'n': 32, 'colors': colors32}
