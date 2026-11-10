def dfs(start_idx, current_indices, current_mask, size):
    if size == target_size:
        found[0] = current_indices[:]
        return True
    if size + (n - start_idx) < target_size:
        return False
    
    # Time check
    if time.time() > deadline - 1.0:
        return False

    for i in range(start_idx, n):
        w = weights[i]
        # Collision check: if any existing sum s has s+w also in sums
        # mask has bit s set. mask << w has bit s+w set.
        # Intersection non-zero means collision.
        if current_mask & (current_mask << w):
            continue
        
        new_mask = current_mask | (current_mask << w)
        current_indices.append(i)
        if dfs(i + 1, current_indices, new_mask, size + 1):
            return True
        current_indices.pop()
        
    return False
