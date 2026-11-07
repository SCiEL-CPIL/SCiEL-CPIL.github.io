def dfs(used_mask, remaining, current_max_idx, chosen_indices):
    nonlocal best_result, best_max_idx
    
    if time.time() > deadline:
        return
    
    if remaining == 0:
        if used_mask == full_mask:
            # The max index is current_max_idx
            if current_max_idx + 1 < best_max_idx:
                best_max_idx = current_max_idx + 1
                best_result = list(chosen_indices)
        return
    
    unused = full_mask ^ used_mask
    if unused.bit_count() < 3 * remaining:
        return
    
    for i in range(len(edge_order)):
        if i + 1 >= best_max_idx:
            break
        
        m = masks[i]
        if m & used_mask:
            continue
        
        new_max_idx = max(current_max_idx, i)
        
        chosen_indices.append(i)
        dfs(used_mask | m, remaining - 1, new_max_idx, chosen_indices)
        chosen_indices.pop()
        
        if best_max_idx == 1:
            return
