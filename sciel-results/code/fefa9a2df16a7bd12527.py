import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def edge_idx(i, j):
        if i > j: i, j = j, i
        return 15*i - i*(i-1)//2 + (j - i - 1)

    k4_edges = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d),
                        edge_idx(c, d)
                    ]
                    k4_edges.append(tuple(edges))
                    
    def get_1_factorization_coloring():
        colors = [0] * 120
        for k in range(15):
            color = k
            colors[edge_idx(15, k)] = color
            for i in range(1, 8):
                u = (k + i) % 15
                v = (k - i) % 15
                colors[edge_idx(u, v)] = color
        return colors

    def get_k4_colors(col_list, k4_idx):
        edges = k4_edges[k4_idx]
        return [col_list[e] for e in edges]

    def count_distinct(col_list, k4_idx):
        return len(set(get_k4_colors(col_list, k4_idx)))

    def find_invalid_k4s(col_list):
        return [i for i in range(len(k4_edges)) if count_distinct(col_list, i) < 5]

    def rebuild_k4s_using(col_list, max_color):
        new_k4s_using = [set() for _ in range(max_color)]
        for k4_idx, edges in enumerate(k4_edges):
            distinct_colors_in_k4 = set()
            for e in edges:
                c = col_list[e]
                distinct_colors_in_k4.add(c)
            for c in distinct_colors_in_k4:
                if c < max_color:
                    new_k4s_using[c].add(k4_idx)
        return new_k4s_using

    def is_valid_subset(k4_indices, col_list):
        for k4_idx in k4_indices:
            if count_distinct(col_list, k4_idx) < 5:
                return False
        return True

    best_colors = get_1_factorization_coloring()
    best_count = len(set(best_colors))
    
    # Repair phase
    # We will iterate and fix invalid K4s
    # Since we reuse colors, the set of active colors remains {0..14}
    # We just change edge assignments
    
    for iteration in range(100):
        if time.time() > deadline - 20: break
        invalid_k4s = find_invalid_k4s(best_colors)
        if not invalid_k4s:
            break
            
        # Pick a random invalid K4
        k4_idx = random.choice(invalid_k4s)
        k4_cols = get_k4_colors(best_colors, k4_idx)
        distinct_cols = set(k4_cols)
        
        if len(distinct_cols) >= 5:
            continue # Should not happen
            
        # Find edges where the color is unique in this K4
        # If we change such an edge to a new color, the count increases by 1
        unique_edges = []
        for e_idx, c in zip(k4_edges[k4_idx], k4_cols):
            if k4_cols.count(c) == 1:
                unique_edges.append(e_idx)
                
        if not unique_edges:
            # If no unique edges, we can't increase count by swapping?
            # Actually, if all colors appear at least twice, then with 6 edges and < 5 colors,
            # we have e.g. 3 colors, each appearing 2 times.
            # If we change one edge of color c to new color c_new,
            # c still appears (1 time), c_new appears (1 time).
            # Count becomes: (old count - 1 if c was unique? No c was not unique)
            # Old distinct: {c1, c2, c3}.
            # New distinct: {c1, c2, c3, c_new}. (c is still there).
            # So count increases by 1!
            # So we can pick any edge.
            unique_edges = list(range(6)) # indices in the k4_edges tuple
            
        e_idx_to_change = random.choice(unique_edges)
        global_e_idx = k4_edges[k4_idx][e_idx_to_change]
        
        # Pick a new color not in distinct_cols
        available_colors = [c for c in range(15) if c not in distinct_cols]
        if available_colors:
            new_c = random.choice(available_colors)
        else:
            # This case shouldn't happen as len(distinct_cols) < 5
            new_c = 15 
            
        best_colors[global_e_idx] = new_c

    best_count = len(set(best_colors))
    
    # Merge phase
    current_colors = best_colors.copy()
    current_active = set(current_colors)
    max_color = 15
    current_k4s_using = rebuild_k4s_using(current_colors, max_color)
    
    for restart in range(50):
        if time.time() > deadline - 5: break
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            cols_list = list(current_active)
            cols_list.sort(key=lambda c: len(current_k4s_using[c]))
            
            merged = False
            for i in range(len(cols_list)):
                c1 = cols_list[i]
                if c1 not in current_active: continue
                
                candidates = [c for c in current_active if c != c1]
                random.shuffle(candidates)
                
                for c2 in candidates:
                    if c2 not in current_active: continue
                    
                    affected = current_k4s_using[c1]
                    
                    new_colors = current_colors.copy()
                    for k in range(120):
                        if new_colors[k] == c1:
                            new_colors[k] = c2
                    
                    if is_valid_subset(affected, new_colors):
                        current_colors = new_colors
                        current_active.remove(c1)
                        current_k4s_using[c2] = current_k4s_using[c2] | current_k4s_using[c1]
                        current_k4s_using[c1] = set()
                        
                        improved = True
                        if len(current_active) < best_count:
                            best_count = len(current_active)
                            best_colors = current_colors.copy()
                        merged = True
                        break
                if merged:
                    break
                    
    return best_colors
