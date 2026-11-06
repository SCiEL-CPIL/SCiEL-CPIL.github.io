"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    # Target n=50 (Chung's record)
    n = 50
    
    # Precompute edges and indices
    edges = []
    edge_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))
    
    num_edges = len(edges)
    
    def get_idx(i, j):
        return edge_to_idx.get((i, j) if i < j else (j, i))
    
    # Local Search / Simulated Annealing
    def local_search(n, max_time=110):
        start_time = time.time()
        
        # Multiple restarts
        for restart in range(50):
            random.seed(restart)
            np.random.seed(restart)
            
            # Initialize random coloring
            colors = np.random.randint(0, 4, size=num_edges)
            
            # Temperature for Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.995
            
            # Helper to calculate delta for changing edge (u,v) to c_new
            def get_delta(u, v, c_new):
                delta = 0
                idx_uv = get_idx(u, v)
                c_old = colors[idx_uv]
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    idx_uw = get_idx(u, w)
                    idx_vw = get_idx(v, w)
                    
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    
                    old_mono = 1 if (c_old == c_uw == c_vw) else 0
                    new_mono = 1 if (c_new == c_uw == c_vw) else 0
                    
                    delta += new_mono - old_mono
                return delta

            # Check if valid
            def is_valid():
                # Iterate all triangles
                for i in range(n):
                    for j in range(i + 1, n):
                        idx_ij = edge_to_idx[(i, j)]
                        c_ij = colors[idx_ij]
                        for k in range(j + 1, n):
                            idx_ik = edge_to_idx[(i, k)]
                            idx_jk = edge_to_idx[(j, k)]
                            if colors[idx_ik] == c_ij and colors[idx_jk] == c_ij:
                                return False
                return True

            steps = 0
            max_steps = 200000
            
            while steps < max_steps:
                if time.time() - start_time > max_time:
                    return None
                
                # Pick a random edge
                idx = np.random.randint(num_edges)
                u, v = edges[idx]
                c_old = colors[idx]
                
                best_delta = 0
                best_color = c_old
                
                # Find best color change
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    delta = get_delta(u, v, c_new)
                    if delta < best_delta:
                        best_delta = delta
                        best_color = c_new
                
                # Acceptance probability
                if best_delta < 0:
                    accept = True
                elif best_delta == 0:
                    accept = random.random() < 0.5
                else:
                    # Simulated Annealing
                    if T > T_min:
                        accept = random.random() < np.exp(-best_delta / T)
                    else:
                        accept = False
                
                if accept:
                    colors[idx] = best_color
                    T *= cooling
                
                # Check validity periodically
                if steps % 100 == 0:
                    if is_valid():
                        return colors.tolist()
                
                steps += 1
        
        return None

    # Try to find n=50
    result = local_search(n, max_time=110)
    if result is not None:
        return {'n': n, 'colors': result}
    
    # Fallback: return the known n=32 construction
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
# EVOLVE-BLOCK-END
