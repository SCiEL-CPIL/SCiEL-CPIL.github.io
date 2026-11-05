# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute K4s and edge-to-K4 mapping
    k4s = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_index_map = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        e_indices = [
            edge_index_map[(v0, v1)],
            edge_index_map[(v0, v2)],
            edge_index_map[(v0, v3)],
            edge_index_map[(v1, v2)],
            edge_index_map[(v1, v3)],
            edge_index_map[(v2, v3)],
        ]
        k4_idx = len(k4s)
        k4s.append(e_indices)
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
    
    num_k4s = len(k4s)
    
    def check_k4_valid(k4_idx, col):
        e_indices = k4s[k4_idx]
        return len({col[i] for i in e_indices}) >= 5

    def initial_state(col):
        """Compute initial violations, color counts, and k4_valid array."""
        k4_valid = [1] * num_k4s
        total_violations = 0
        for k4_idx in range(num_k4s):
            if not check_k4_valid(k4_idx, col):
                k4_valid[k4_idx] = 0
                total_violations += 1
        
        color_counts = [0] * 120 # Max color index is 119
        for c in col:
            color_counts[c] += 1
        
        current_colors = sum(1 for c in color_counts if c > 0)
        
        return total_violations, current_colors, k4_valid, color_counts

    best_result = None
    best_color_count = float('inf')
    
    # Fallback seed
    seed_col = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    v, c, _, _ = initial_state(seed_col)
    if v == 0:
        best_result = seed_col
        best_color_count = c
    
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 20.0:
            break
        
        for restart in range(50):
            if time.time() > deadline - 10.0:
                break
            
            col = [random.randint(0, target_colors - 1) for _ in range(num_edges)]
            current_violations, current_colors, k4_valid, color_counts = initial_state(col)
            
            if current_violations == 0:
                if current_colors < best_color_count:
                    best_color_count = current_colors
                    best_result = col[:]
                break
            
            no_improve = 0
            max_iters = 50000 # Increased iterations due to speed
            
            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    break
                
                e = random.randint(0, num_edges - 1)
                old_color = col[e]
                new_color = random.randint(0, target_colors - 1)
                
                if new_color == old_color:
                    continue
                
                # Update color counts
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    current_colors -= 1
                color_counts[new_color] += 1
                if color_counts[new_color] == 1:
                    current_colors += 1
                
                # Update K4 validities
                affected_k4s = edge_to_k4s[e]
                was_valid = [k4_valid[k] for k in affected_k4s]
                
                col[e] = new_color
                
                delta_violations = 0
                for i, k4_idx in enumerate(affected_k4s):
                    is_valid = 1 if check_k4_valid(k4_idx, col) else 0
                    k4_valid[k4_idx] = is_valid
                    if was_valid[i] == 1 and is_valid == 0:
                        delta_violations += 1
                    elif was_valid[i] == 0 and is_valid == 1:
                        delta_violations -= 1
                
                new_violations = current_violations + delta_violations
                
                # Acceptance criterion
                if new_violations < current_violations:
                    current_violations = new_violations
                    no_improve = 0
                elif new_violations == current_violations:
                    if current_violations == 0 and current_colors < best_color_count:
                        best_color_count = current_colors
                        best_result = col[:]
                        break
                    # Prefer fewer colors if same violations
                    # We already updated current_colors
                    if random.random() < 0.5:
                        current_violations = new_violations
                        no_improve = 0
                    else:
                        # Revert
                        col[e] = old_color
                        color_counts[old_color] -= 1
                        if color_counts[old_color] == 0:
                            current_colors -= 1
                        color_counts[new_color] += 1
                        if color_counts[new_color] == 1:
                            current_colors += 1
                        for i, k4_idx in enumerate(affected_k4s):
                            k4_valid[k4_idx] = was_valid[i]
                else:
                    # Revert
                    col[e] = old_color
                    color_counts[old_color] -= 1
                    if color_counts[old_color] == 0:
                        current_colors -= 1
                    color_counts[new_color] += 1
                    if color_counts[new_color] == 1:
                        current_colors += 1
                    for i, k4_idx in enumerate(affected_k4s):
                        k4_valid[k4_idx] = was_valid[i]
                
                if current_violations == 0:
                    if current_colors < best_color_count:
                        best_color_count = current_colors
                        best_result = col[:]
                    break
                
                if no_improve > 1000:
                    break
            
            if current_violations == 0:
                if current_colors < best_color_count:
                    best_color_count = current_colors
                    best_result = col[:]
                break
    
    if best_result is not None:
        return best_result
    
    return seed_col
# EVOLVE-BLOCK-END
