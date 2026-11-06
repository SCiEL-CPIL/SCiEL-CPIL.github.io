import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    vertices = list(range(2, 257))
    n = len(vertices)
    val_to_idx = {v: i for i, v in enumerate(vertices)}
    weights = [1.0/v for v in vertices]
    
    # Pre-sort indices by weight descending (value ascending)
    sorted_indices = sorted(range(n), key=lambda i: vertices[i])
    
    neighbors = [0] * n
    for i in range(n):
        vi = vertices[i]
        for j in range(i+1, n):
            vj = vertices[j]
            if lcm(vi, vj) <= 256:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
                
    # Initial Greedy
    used_mask = 0
    selected_indices = []
    for i in sorted_indices:
        if (neighbors[i] & used_mask) == 0:
            selected_indices.append(i)
            used_mask |= (1 << i)
            
    best_sum = sum(weights[i] for i in selected_indices)
    best_set = set(selected_indices)
    
    # B&B Setup
    best_sum_global = best_sum
    best_set_global = best_set.copy()
    node_count = 0
    
    def get_bound(candidates_mask):
        color_classes = []
        for i in sorted_indices:
            if not (candidates_mask >> i) & 1:
                continue
            bit_i = 1 << i
            placed = False
            for c_idx, (max_w, class_mask) in enumerate(color_classes):
                if (neighbors[i] & class_mask) == class_mask:
                    new_max_w = max(max_w, weights[i])
                    color_classes[c_idx] = (new_max_w, class_mask | bit_i)
                    placed = True
                    break
            if not placed:
                color_classes.append((weights[i], bit_i))
        return sum(w for w, m in color_classes)

    import sys
    sys.setrecursionlimit(3000)
    
    def bb(candidates_mask, current_sum, current_set):
        nonlocal best_sum_global, best_set_global, node_count
        
        node_count += 1
        if node_count % 1000 == 0:
            if time.time() > deadline:
                return
                
        if candidates_mask == 0:
            if current_sum > best_sum_global:
                best_sum_global = current_sum
                best_set_global = current_set.copy()
            return
            
        bound = get_bound(candidates_mask)
        if current_sum + bound <= best_sum_global:
            return
            
        # Find branch vertex (highest weight in candidates)
        branch_idx = -1
        for i in sorted_indices:
            if (candidates_mask >> i) & 1:
                branch_idx = i
                break
                
        if branch_idx == -1:
            return
            
        bit_b = 1 << branch_idx
        
        # Branch 1: Include
        new_mask_include = candidates_mask & ~(neighbors[branch_idx] | bit_b)
        new_set_include = current_set.copy()
        new_set_include.add(branch_idx)
        bb(new_mask_include, current_sum + weights[branch_idx], new_set_include)
        
        # Branch 2: Exclude
        new_mask_exclude = candidates_mask & ~bit_b
        bb(new_mask_exclude, current_sum, current_set)
        
    init_mask = (1 << n) - 1
    init_set = set()
    
    bb(init_mask, 0.0, init_set)
    
    result_values = [vertices[i] for i in best_set_global]
    
    # Ensure non-empty
    if not result_values:
        return [256]
        
    return result_values
