import time
import random
import sys

def construct():
    # 1. Construct the n=32 base coloring
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
        
    def color_32(i, j):
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        else:
            return 3

    n_base = 32
    colors_base = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors_base[idx] = color_32(i, j)
            idx += 1
            
    # Verify base validity (optional, but good for safety)
    # Assuming it's valid as per seed history.

    n_target = 50
    
    # Precompute index map for n=50
    idx_map = {}
    idx = 0
    for i in range(n_target):
        for j in range(i + 1, n_target):
            idx_map[(i, j)] = idx
            idx += 1
            
    def get_idx(i, j):
        if i > j: i, j = j, i
        return idx_map[(i, j)]

    # Initialize full array
    colors = [0] * (n_target * (n_target - 1) // 2)
    
    # Set base colors
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors[get_idx(i, j)] = color_32(i, j)

    # Helper to count mono triangles involving a specific vertex v
    # Triangles are (u, w, v) with u, w < v
    def count_conflicts_for_vertex(v, cols):
        count = 0
        for u in range(v):
            c_uv = cols[get_idx(u, v)]
            for w in range(u):
                c_uw = cols[get_idx(u, w)]
                c_wv = cols[get_idx(w, v)]
                if c_uv == c_uw == c_wv:
                    count += 1
        return count

    best_valid_colors = colors[:]
    best_valid_n = n_base

    # Greedy vertex addition with Min-Conflicts
    for v in range(n_base, n_target):
        # Initialize edges (u, v) for u in 0..v-1
        for u in range(v):
            colors[get_idx(u, v)] = random.randint(0, 3)
            
        # Min-Conflicts local search for this vertex
        solved = False
        max_iters = 3000
        for _ in range(max_iters):
            if count_conflicts_for_vertex(v, colors) == 0:
                solved = True
                break
            
            u = random.randint(0, v - 1)
            old_c = colors[get_idx(u, v)]
            best_c = old_c
            best_conf = count_conflicts_for_vertex(v, colors)
            
            for new_c in range(4):
                if new_c == old_c: continue
                colors[get_idx(u, v)] = new_c
                new_conf = count_conflicts_for_vertex(v, colors)
                if new_conf < best_conf:
                    best_conf = new_conf
                    best_c = new_c
            colors[get_idx(u, v)] = best_c
            
        if solved:
            best_valid_colors = colors[:]
            best_valid_n = v + 1
        else:
            # Try a few random restarts
            for restart in range(5):
                for u in range(v):
                    colors[get_idx(u, v)] = random.randint(0, 3)
                for _ in range(1000):
                    if count_conflicts_for_vertex(v, colors) == 0:
                        solved = True
                        break
                    u = random.randint(0, v - 1)
                    old_c = colors[get_idx(u, v)]
                    best_c = old_c
                    best_conf = count_conflicts_for_vertex(v, colors)
                    for new_c in range(4):
                        if new_c == old_c: continue
                        colors[get_idx(u, v)] = new_c
                        new_conf = count_conflicts_for_vertex(v, colors)
                        if new_conf < best_conf:
                            best_conf = new_conf
                            best_c = new_c
                    colors[get_idx(u, v)] = best_c
                if solved:
                    best_valid_colors = colors[:]
                    best_valid_n = v + 1
                    break
            if not solved:
                # Failed to add vertex v. Break.
                # We have a valid coloring for best_valid_n
                break

    if best_valid_n == n_target:
        return {'n': n_target, 'colors': best_valid_colors}

    # If we didn't reach 50, try to extend the best valid state using SA
    # We need to add vertices from best_valid_n to 49
    # This is complex. Let's just return the best valid n found.
    # Or, run SA on the best_valid_n graph to see if we can fix conflicts if we try to add more?
    # No, if we break, the graph is valid for best_valid_n.
    
    # Let's try to run SA on the *full* n=50 graph initialized with the best valid prefix + random suffix
    # This might allow jumping to 50.
    
    # Initialize colors for SA based on best_valid_colors
    sa_colors = best_valid_colors[:]
    for v in range(best_valid_n, n_target):
        for u in range(v):
            sa_colors[get_idx(u, v)] = random.randint(0, 3)
            
    # Precompute triangles for SA
    num_edges = n_target * (n_target - 1) // 2
    edge_triangles = [[] for _ in range(num_edges)]
    for u in range(n_target):
        for v in range(u + 1, n_target):
            idx_uv = get_idx(u, v)
            for w in range(n_target):
                if w == u or w == v:
                    continue
                idx_uw = get_idx(u, w)
                idx_vw = get_idx(v, w)
                edge_triangles[idx_uv].append((idx_uw, idx_vw))

    def count_mono_triangles(cols):
        count = 0
        for e in range(num_edges):
            c_e = cols[e]
            for idx_uw, idx_vw in edge_triangles[e]:
                if cols[idx_uw] == c_e and cols[idx_vw] == c_e:
                    count += 1
        return count // 3

    current_colors = sa_colors[:]
    current_mono = count_mono_triangles(current_colors)
    best_sa_colors = current_colors[:]
    best_mono = current_mono
    
    start_time = time.time()
    timeout = 110.0
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        e_idx = random.randint(0, num_edges - 1)
        old_c = current_colors[e_idx]
        new_c = random.choice([c for c in range(4) if c != old_c])
        
        delta = 0
        for idx_uw, idx_vw in edge_triangles[e_idx]:
            c_uw = current_colors[idx_uw]
            c_vw = current_colors[idx_vw]
            
            if c_uw == old_c and c_vw == old_c:
                delta -= 1
            if c_uw == new_c and c_vw == new_c:
                delta += 1
                
        new_mono = current_mono + delta
        
        if new_mono < current_mono or random.random() < pow(2.718281828, -delta / T):
            current_colors[e_idx] = new_c
            current_mono = new_mono
            if current_mono < best_mono:
                best_mono = current_mono
                best_sa_colors = current_colors[:]
                if best_mono == 0:
                    break
        T *= cooling
        
    if best_mono == 0:
        return {'n': n_target, 'colors': best_sa_colors}
    else:
        # Return the best valid n found during greedy phase
        # We need to extract the colors for best_valid_n from best_valid_colors
        # best_valid_colors is a full n=50 array, but only edges < best_valid_n are guaranteed valid.
        # We should return a truncated array.
        
        num_edges_best = best_valid_n * (best_valid_n - 1) // 2
        # Extract colors in lexicographic order for n=best_valid_n
        final_colors = [0] * num_edges_best
        idx = 0
        for i in range(best_valid_n):
            for j in range(i + 1, best_valid_n):
                final_colors[idx] = best_valid_colors[get_idx(i, j)]
                idx += 1
                
        return {'n': best_valid_n, 'colors': final_colors}

# EVOLVE-BLOCK-END
