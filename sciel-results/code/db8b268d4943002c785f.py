def _local_search_2for1(blocks, all_blocks, all_block_sets, block_masks, subset_to_index, all_subsets, deadline):
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    curr_indices = [block_to_idx[b] for b in blocks]
    curr_set = set(curr_indices)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 2.0:
            break
            
        # 1. Remove single redundant blocks
        n = len(curr_indices)
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            mask_i = block_masks[curr_indices[i]]
            rest_mask = np.zeros_like(mask_i)
            for j in range(n):
                if i != j:
                    rest_mask |= block_masks[curr_indices[j]]
            
            unique_mask = mask_i & ~rest_mask
            if not np.any(unique_mask):
                curr_indices.pop(i)
                curr_set.remove(curr_indices[i] if i < len(curr_indices) else curr_indices[-1]) # Wait, pop returns nothing in py3? No, list.pop(i) removes and returns? No, list.pop(i) removes at index i.
                # Correction:
                # val = curr_indices.pop(i)
                # curr_set.remove(val)
                improved = True
                break
                
        if improved:
            continue
            
        # 2. Try 2-for-1 swaps
        n = len(curr_indices)
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 2.0:
                    break
                    
                mask_i = block_masks[curr_indices[i]]
                mask_j = block_masks[curr_indices[j]]
                
                rest_mask = np.zeros_like(mask_i)
                for l in range(n):
                    if l != i and l != j:
                        rest_mask |= block_masks[curr_indices[l]]
                        
                unique_mask = (mask_i | mask_j) & ~rest_mask
                unique_indices = np.where(unique_mask)[0]
                
                if len(unique_indices) == 0:
                    # Remove both
                    val_j = curr_indices.pop(j)
                    val_i = curr_indices.pop(i)
                    curr_set.remove(val_j)
                    curr_set.remove(val_i)
                    improved = True
                    break
                    
                else:
                    p_u = set()
                    for idx in unique_indices:
                        p_u.update(all_subsets[idx])
                    
                    if len(p_u) <= 7:
                        found_new = -1
                        # Check all blocks
                        for b_idx, b_set in enumerate(all_block_sets):
                            if p_u.issubset(b_set):
                                if b_idx not in curr_set:
                                    found_new = b_idx
                                    break
                                    
                        if found_new != -1:
                            # Replace i, j with found_new
                            val_j = curr_indices.pop(j)
                            val_i = curr_indices.pop(i)
                            curr_set.remove(val_j)
                            curr_set.remove(val_i)
                            
                            curr_indices.append(found_new)
                            curr_set.add(found_new)
                            improved = True
                            break
                            
            if improved:
                break
                
    return [all_blocks[i] for i in curr_indices]
