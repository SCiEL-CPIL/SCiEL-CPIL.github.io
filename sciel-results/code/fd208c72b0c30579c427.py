def remove_redundant(blocks, M, num_t_subsets):
    if not blocks:
        return []
    
    counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b in blocks:
        indices = M[b].nonzero()[1]
        counts[indices] += 1
        
    changed = True
    while changed:
        changed = False
        redundant_indices = []
        for i, b in enumerate(blocks):
            indices = M[b].nonzero()[1]
            if np.all(counts[indices] > 1):
                redundant_indices.append(i)
                
        if redundant_indices:
            changed = True
            # Remove all redundant at once
            # Update counts
            for i in redundant_indices:
                b = blocks[i]
                indices = M[b].nonzero()[1]
                counts[indices] -= 1
            
            # Filter blocks
            redundant_set = set(redundant_indices)
            blocks = [b for i, b in enumerate(blocks) if i not in redundant_set]
            
    return blocks
