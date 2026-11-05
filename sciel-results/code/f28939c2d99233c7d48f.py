import time
import random

def construct():
    deadline = time.time() + 115

    # --- Base n=32 colouring (Greenwood-Gleason) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def edge_color_core(i, j):
        return log_map[i ^ j] % 3

    n_base = 32
    base_colors = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                c = edge_color_core(i % 16, j % 16)
            else:
                c = 3
            base_colors[idx] = c
            idx += 1

    # --- Local Search for n=50 ---
    N = 50
    num_edges = N * (N - 1) // 2
    
    # Edge mapping
    edges = []
    edge_to_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))
            
    # Precompute neighbors for each edge
    edge_neighbors = [[] for _ in range(num_edges)]
    
    for idx, (u, v) in enumerate(edges):
        for w in range(N):
            if w == u or w == v:
                continue
            if u < w:
                idx_uw = edge_to_idx[(u, w)]
            else:
                idx_uw = edge_to_idx[(w, u)]
                
            if v < w:
                idx_vw = edge_to_idx[(v, w)]
            else:
                idx_vw = edge_to_idx[(w, v)]
                
            edge_neighbors[idx].append((w, idx_uw, idx_vw))

    def init_mono_list(colors):
        mono_list = []
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                idx_ij = edge_to_idx[(i, j)]
                c_ij = colors[idx_ij]
                for k in range(j + 1, N):
                    idx_jk = edge_to_idx[(j, k)]
                    idx_ik = edge_to_idx[(i, k)]
                    if colors[idx_jk] == c_ij and colors[idx_ik] == c_ij:
                        mono_list.append((i, j, k))
        return mono_list

    def update_mono_list(mono_list, colors, edge_idx, old_c, new_c):
        u, v = edges[edge_idx]
        to_remove = set()
        to_add = []
        
        for w, idx_uw, idx_vw in edge_neighbors[edge_idx]:
            verts = sorted([u, v, w])
            tri_key = tuple(verts)
            
            c_uw = colors[idx_uw]
            c_vw = colors[idx_vw]
            
            was_mono = (old_c == c_uw == c_vw)
            is_mono_new = (new_c == c_uw == c_vw)
            
            if was_mono and not is_mono_new:
                to_remove.add(tri_key)
            elif is_mono_new and not was_mono:
                to_add.append(tri_key)
        
        if not to_remove and not to_add:
            return mono_list
        
        new_mono_list = []
        for tri in mono_list:
            if tri not in to_remove:
                new_mono_list.append(tri)
        new_mono_list.extend(to_add)
        return new_mono_list

    best_n = n_base
    best_colors = base_colors[:]
    
    attempts = 0
    max_attempts = 50
    while time.time() < deadline and attempts < max_attempts:
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        mono_list = init_mono_list(colors)
        
        if len(mono_list) == 0:
            return {'n': N, 'colors': colors}
            
        temperature = 2.0
        min_temp = 0.01
        cooling = 0.9997
        steps = 0
        max_steps = 200000
        
        best_mono_count = len(mono_list)
        no_improve_steps = 0
        no_improve_limit = 5000
        
        while len(mono_list) > 0 and steps < max_steps and time.time() < deadline:
            if len(mono_list) == 0:
                break
                
            tri_idx = random.randint(0, len(mono_list) - 1)
            i, j, k = mono_list[tri_idx]
            
            edge_choice = random.randint(0, 2)
            if edge_choice == 0:
                u, v = i, j
            elif edge_choice == 1:
                u, v = j, k
            else:
                u, v = i, k
                
            idx_uv = edge_to_idx[(min(u, v), max(u, v))]
            old_c = colors[idx_uv]
            
            best_delta = float('inf')
            best_new_c = -1
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                created = 0
                destroyed = 0
                
                for w, idx_uw, idx_vw in edge_neighbors[idx_uv]:
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    
                    was_mono = (old_c == c_uw == c_vw)
                    is_mono_new = (new_c == c_uw == c_vw)
                    
                    if was_mono:
                        destroyed += 1
                    if is_mono_new:
                        created += 1
                        
                delta = created - destroyed
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
                    
            if best_new_c == -1:
                break
                
            accept = False
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                if random.random() < 0.3:
                    accept = True
            else:
                if random.random() < pow(2.71828, -best_delta / temperature):
                    accept = True
            
            if accept:
                colors[idx_uv] = best_new_c
                mono_list = update_mono_list(mono_list, colors, idx_uv, old_c, best_new_c)
                steps += 1
                
                if len(mono_list) < best_mono_count:
                    best_mono_count = len(mono_list)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                if no_improve_steps > no_improve_limit and temperature < 1.0:
                    temperature = 1.0
                    no_improve_steps = 0
                    no_improve_limit = int(no_improve_limit * 1.5)
            
            temperature *= cooling
            if temperature < min_temp:
                temperature = min_temp

        if len(mono_list) == 0:
            return {'n': N, 'colors': colors}
        
        attempts += 1

    return {'n': best_n, 'colors': best_colors}
