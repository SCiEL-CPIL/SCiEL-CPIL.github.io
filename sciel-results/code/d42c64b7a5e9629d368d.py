"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Fallback: 32-vertex Greenwood-Gleason coloring
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def base_colors_32():
        n = 32
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return {'n': n, 'colors': colors}

    best_result = base_colors_32()
    
    # Target: 50 vertices
    N = 50
    num_edges = N * (N - 1) // 2
    
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (2 * N - i - 1) // 2 + (j - i - 1)

    def count_mono_triangles(colors):
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                c_ij = colors[get_idx(i, j)]
                for k in range(j + 1, N):
                    c_ik = colors[get_idx(i, k)]
                    c_jk = colors[get_idx(j, k)]
                    if c_ij == c_ik == c_jk:
                        count += 1
        return count

    def local_search():
        # Random initialization
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        current_count = count_mono_triangles(colors)
        
        if current_count == 0:
            return colors[:]
            
        # Precompute triangle lists for each edge to speed up delta calculation
        # tri_list[e] = list of (other_edge1, other_edge2, vertex)
        # Actually, for edge (u,v), the triangles are (u,v,w).
        # The other edges are (u,w) and (v,w).
        
        # To be faster, we can just iterate w from 0 to N-1.
        
        stuck_counter = 0
        max_stuck = 10000
        
        while time.time() < deadline:
            if current_count == 0:
                return colors[:]
                
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            
            # Determine u, v from e?
            # It's easier to pick random u, v.
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            e_idx = get_idx(u, v)
            c_old = colors[e_idx]
            
            best_delta = 0
            best_new_color = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                    
                delta = 0
                for w in range(N):
                    if w == u or w == v:
                        continue
                    
                    # Triangle (u, v, w)
                    # Edges: (u,v) [changing], (u,w), (v,w)
                    idx_uw = get_idx(u, w)
                    idx_vw = get_idx(v, w)
                    
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    
                    # Old mono status
                    old_mono = 1 if (c_old == c_uw == c_vw) else 0
                    # New mono status
                    new_mono = 1 if (c_new == c_uw == c_vw) else 0
                    
                    delta += (new_mono - old_mono)
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = c_new
                    
            if best_new_color != -1:
                colors[e_idx] = best_new_color
                current_count += best_delta
                stuck_counter = 0
            else:
                # No improving move found for this edge
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    # Random restart or perturbation
                    # Perturb: flip 10 random edges
                    for _ in range(10):
                        ru = random.randint(0, N - 1)
                        rv = random.randint(0, N - 1)
                        if ru != rv:
                            if ru > rv:
                                ru, rv = rv, ru
                            re_idx = get_idx(ru, rv)
                            colors[re_idx] = random.randint(0, 3)
                    current_count = count_mono_triangles(colors)
                    stuck_counter = 0
                    max_stuck = 10000
                    
        return None

    # Run local search with restarts
    # We might need multiple restarts if the first one gets stuck in a deep local minimum
    # But usually, random init + local search works well for N=50.
    
    found = local_search()
    if found is not None:
        return {'n': N, 'colors': found}
    
    # If local search failed, try a few more restarts if time permits
    while time.time() < deadline:
        found = local_search()
        if found is not None:
            return {'n': N, 'colors': found}
            
    return best_result

# EVOLVE-BLOCK-END
