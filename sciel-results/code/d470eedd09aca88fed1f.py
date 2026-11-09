    chosen_indices = []
    
    def branch_and_bound(available_mask, current_weight, avail_sum):
        nonlocal best_weight, best_solution
        
        if time.time() > deadline:
            return
        
        ub = current_weight + avail_sum
        if ub <= best_weight + 1e-12:
            return
        
        if available_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_solution = list(chosen_indices)
            return
        
        best_i = -1
        m = available_mask
        i = 0
        while m:
            if m & 1:
                best_i = i
                break
            m >>= 1
            i += 1
        
        if best_i == -1:
            return
        
        # Branch 1: Include best_i
        remove_mask = (1 << best_i) | conflict_mask[best_i]
        new_available_include = available_mask & ~remove_mask
        
        removed_sum = 0.0
        rm = available_mask & remove_mask
        idx = 0
        while rm:
            if rm & 1:
                removed_sum += scores[nodes[idx]]
            rm >>= 1
            idx += 1
        
        new_avail_sum_include = avail_sum - removed_sum
        
        chosen_indices.append(best_i)
        branch_and_bound(new_available_include, current_weight + scores[nodes[best_i]], new_avail_sum_include)
        chosen_indices.pop()
        
        # Branch 2: Exclude best_i
        new_available_exclude = available_mask & ~(1 << best_i)
        new_avail_sum_exclude = avail_sum - scores[nodes[best_i]]
        
        branch_and_bound(new_available_exclude, current_weight, new_avail_sum_exclude)
