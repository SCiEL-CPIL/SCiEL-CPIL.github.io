import random
import time
import itertools

def construct():
    N = 16
    E = N * (N - 1) // 2
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    k4s = list(itertools.combinations(range(N), 4))
    k4_edges = []
    k4_edge_pos = []
    for k4 in k4s:
        e_idx = []
        pos_dict = {}
        for i in range(4):
            for j in range(i+1, 4):
                idx = edge_to_idx[(k4[i], k4[j])]
                pos_dict[idx] = len(e_idx)
                e_idx.append(idx)
        k4_edges.append(e_idx)
        k4_edge_pos.append(pos_dict)
        
    edge_k4s = [[] for _ in range(E)]
    for k4_idx, e_idx in enumerate(k4_edges):
        for e in e_idx:
            edge_k4s[e].append(k4_idx)
            
    # Precompute K4 colour counts structure
    # k4_counts[k4_idx] = {colour: count}
    # k4_distinct[k4_idx] = number of distinct colours
    
    def init_counts(colors):
        k4_counts = [{} for _ in range(len(k4s))]
        k4_distinct = [0] * len(k4s)
        bad_count = 0
        for k4_idx, e_idx in enumerate(k4_edges):
            counts = {}
            for e in e_idx:
                c = colors[e]
                counts[c] = counts.get(c, 0) + 1
            k4_counts[k4_idx] = counts
            k4_distinct[k4_idx] = len(counts)
            if k4_distinct[k4_idx] < 5:
                bad_count += 1
        return k4_counts, k4_distinct, bad_count

    def get_delta_bad(k4_indices, k4_counts, k4_distinct, edge_idx, new_color, old_color):
        delta = 0
        if old_color == new_color:
            return 0
        for k4_idx in k4_indices:
            counts = k4_counts[k4_idx]
            
            # Update counts for old_color
            c_old = counts[old_color]
            if c_old == 1:
                # Colour disappears
                k4_distinct[k4_idx] -= 1
                del counts[old_color]
            else:
                counts[old_color] = c_old - 1
                
            # Update counts for new_color
            c_new = counts.get(new_color, 0)
            if c_new == 0:
                # New colour appears
                k4_distinct[k4_idx] += 1
                counts[new_color] = 1
            else:
                counts[new_color] = c_new + 1
                
            # Update delta
            # We need to know if the K4 status changed
            # Status: bad if distinct < 5
            # We need the distinct count BEFORE the change to compare?
            # No, we can just check the current distinct count vs 5.
            # But we need to know if it WAS bad and NOW is good, or vice versa.
            # We can store the distinct count before the change in a temporary variable?
            # Or we can just recalculate the delta based on the change in distinct count.
            
            # Let's trace:
            # d_before = k4_distinct[k4_idx] (before update)
            # d_after = k4_distinct[k4_idx] (after update)
            # But we updated k4_distinct in place.
            # So we need to save d_before.
            
        # This approach requires saving state or careful delta calculation.
        # Let's refine:
        
        delta = 0
        for k4_idx in k4_indices:
            counts = k4_counts[k4_idx]
            d_before = k4_distinct[k4_idx]
            
            # Apply change
            c_old = counts[old_color]
            if c_old == 1:
                k4_distinct[k4_idx] -= 1
                del counts[old_color]
            else:
                counts[old_color] = c_old - 1
                
            c_new = counts.get(new_color, 0)
            if c_new == 0:
                k4_distinct[k4_idx] += 1
                counts[new_color] = 1
            else:
                counts[new_color] = c_new + 1
                
            d_after = k4_distinct[k4_idx]
            
            if d_before < 5 and d_after >= 5:
                delta -= 1
            elif d_before >= 5 and d_after < 5:
                delta += 1
                
        return delta

    def local_search(palette_size, time_limit):
        start_time = time.time()
        best_bad = float('inf')
        best_colors = None
        
        for restart in range(5):
            colors = [random.randint(0, palette_size - 1) for _ in range(E)]
            k4_counts, k4_distinct, bad = init_counts(colors)
            
            if bad == 0:
                return colors
            
            current_bad = bad
            current_colors = colors[:]
            
            T = 1.0
            T_min = 0.01
            alpha = 0.9995
            
            while time.time() - start_time < time_limit:
                e_idx = random.randint(0, E - 1)
                old_c = current_colors[e_idx]
                new_c = random.randint(0, palette_size - 1)
                if new_c == old_c:
                    continue
                
                delta = get_delta_bad(edge_k4s[e_idx], k4_counts, k4_distinct, e_idx, new_c, old_c)
                
                if delta < 0 or (delta > 0 and random.random() < pow(2.718, -delta / T)):
                    current_colors[e_idx] = new_c
                    current_bad += delta
                    
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_colors = current_colors[:]
                        
                    if best_bad == 0:
                        return best_colors
                        
                T *= alpha
                if T < T_min:
                    break
                    
        return best_colors

    deadline = time.time() + 115
    
    for p in [13, 14, 15]:
        if time.time() > deadline:
            break
        limit = 20 if p == 13 else 30
        if time.time() + limit > deadline:
            limit = deadline - time.time()
            if limit <= 0: break
            
        colors = local_search(p, limit)
        
        if colors is not None:
            # Verify
            k4_counts, k4_distinct, bad = init_counts(colors)
            if bad == 0:
                return colors
            
    if time.time() < deadline:
        colors = local_search(16, deadline - time.time() - 1)
        if colors is not None:
            k4_counts, k4_distinct, bad = init_counts(colors)
            if bad == 0:
                return colors
            
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
