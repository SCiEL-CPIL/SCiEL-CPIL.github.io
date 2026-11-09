import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    num_edges = len(edges)
    
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
        
    num_k4s = len(k4s)
    
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)

    max_colors = 20
    
    def check_validity_and_count(colors):
        # Returns (is_valid, num_colors)
        k4_counts = [[0] * max_colors for _ in range(num_k4s)]
        k4_mask = [0] * num_k4s
        for k4_idx, e_list in enumerate(k4s):
            mask = 0
            for ei in e_list:
                c = colors[ei]
                k4_counts[k4_idx][c] += 1
                mask |= (1 << c)
            k4_mask[k4_idx] = mask
        
        valid = True
        for mask in k4_mask:
            if mask.bit_count() < 5:
                valid = False
                break
        
        return valid, len(set(colors))

    # Try simple formulas
    for p in [13, 14, 15, 16, 17]:
        cols = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            cols[idx] = (i + j) % p
        valid, nc = check_validity_and_count(cols)
        if valid:
            # Compact and return
            unique = sorted(set(cols))
            mapping = {c: i for i, c in enumerate(unique)}
            return [mapping[c] for c in cols]

    # If none worked, use local search starting from 17 (known valid)
    # Actually, if 17 is valid, we found it above.
    # So if we are here, 13-17 simple formulas failed?
    # Unlikely 17 fails.
    # Let's assume we have a valid 17-coloring.
    
    # Start with 17
    init_cols = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        init_cols[idx] = (i + j) % 17
        
    # Local search to reduce colors
    current_colors = init_cols[:]
    current_kc = [[0] * max_colors for _ in range(num_k4s)]
    current_km = [0] * num_k4s
    for k4_idx, e_list in enumerate(k4s):
        mask = 0
        for ei in e_list:
            c = current_colors[ei]
            current_kc[k4_idx][c] += 1
            mask |= (1 << c)
        current_km[k4_idx] = mask
            
    current_nc = len(set(current_colors))
    best_colors = current_colors[:]
    best_nc = current_nc

    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        e_idx = random.randint(0, num_edges - 1)
        old_c = current_colors[e_idx]
        
        # Try to pick a new color that exists
        # If we pick a new color not in use, we increase nc.
        # We want to decrease nc.
        # So we should try to recolor to an existing color.
        
        existing_colors = list(set(current_colors))
        if not existing_colors:
            break
            
        new_c = random.choice(existing_colors)
        if new_c == old_c:
            continue
            
        # Check validity
        valid_move = True
        for k4_idx in edge_k4s[e_idx]:
            old_count_old_c = current_kc[k4_idx][old_c]
            old_count_new_c = current_kc[k4_idx][new_c]
            
            new_count_old_c = old_count_old_c - 1
            new_count_new_c = old_count_new_c + 1
            
            old_mask = current_km[k4_idx]
            new_mask = old_mask
            if new_count_old_c == 0:
                new_mask &= ~(1 << old_c)
            if new_count_new_c == 1:
                new_mask |= (1 << new_c)
                
            if new_mask.bit_count() < 5:
                valid_move = False
                break
        
        if valid_move:
            # Apply
            current_colors[e_idx] = new_c
            for k4_idx in edge_k4s[e_idx]:
                current_kc[k4_idx][old_c] -= 1
                current_kc[k4_idx][new_c] += 1
                old_mask = current_km[k4_idx]
                new_mask = old_mask
                if current_kc[k4_idx][old_c] == 0:
                    new_mask &= ~(1 << old_c)
                if current_kc[k4_idx][new_c] == 1:
                    new_mask |= (1 << new_c)
                current_km[k4_idx] = new_mask
            
            current_nc = len(set(current_colors))
            if current_nc < best_nc:
                best_nc = current_nc
                best_colors = current_colors[:]
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Compact
    unique = sorted(set(best_colors))
    mapping = {c: i for i, c in enumerate(unique)}
    return [mapping[c] for c in best_colors]
