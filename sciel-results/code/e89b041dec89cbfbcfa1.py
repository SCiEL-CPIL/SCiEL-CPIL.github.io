import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # Precompute block_indices: shape (num_blocks, t_subsets_per_block)
    t_subsets_per_block = 35  # C(7,3) = 35
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(blocks):
        for j, t_sub in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = t_subset_to_idx[t_sub]
    
    # Precompute frequency of each t-subset
    freq = np.bincount(block_indices.flatten(), minlength=num_t_subsets)
    
    # Inverse frequency for weighting
    inv_freq = 1.0 / (freq + 1e-9)
    
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_blocks = []
    
    # Phase 1: Weighted greedy
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        w = np.where(uncovered, inv_freq, 0.0)
        gains = np.sum(w[block_indices], axis=1)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        selected_blocks.append(list(blocks[best_idx]))
        uncovered[block_indices[best_idx]] = False
    
    # Phase 2: Local search - try to swap blocks to reduce count
    if time.time() < deadline - 30:
        selected_blocks = _local_search(selected_blocks, blocks, block_indices, t_subsets, t_subset_to_idx, deadline)
    
    return selected_blocks

def _local_search(selected_blocks, blocks, block_indices, t_subsets, t_subset_to_idx, deadline):
    """Try to reduce the number of blocks by swapping or removing redundant blocks."""
    # Build coverage count for each t-subset
    num_t_subsets = len(t_subsets)
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    
    for block in selected_blocks:
        for t_sub in itertools.combinations(block, 3):
            coverage[t_subset_to_idx[t_sub]] += 1
    
    # Try removing redundant blocks (blocks that don't uniquely cover any t-subset)
    removed_any = True
    while removed_any and time.time() < deadline - 10:
        removed_any = False
        new_selected = []
        for i, block in enumerate(selected_blocks):
            # Check if this block is the only one covering any t-subset
            is_redundant = True
            for t_sub in itertools.combinations(block, 3):
                idx = t_subset_to_idx[t_sub]
                if coverage[idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_selected.append(block)
            else:
                # Remove this block
                for t_sub in itertools.combinations(block, 3):
                    coverage[t_subset_to_idx[t_sub]] -= 1
                removed_any = True
        selected_blocks = new_selected
    
    # Try swap-based local search: replace one block with another that covers more uncovered
    # Actually, after redundancy removal, all t-subsets should still be covered
    # Try to find a better solution by trying alternative blocks
    
    # For each selected block, try replacing it with a different block
    # that maintains coverage but allows further reduction
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        n = len(selected_blocks)
        for i in range(n):
            if time.time() > deadline - 10:
                break
            current_block = selected_blocks[i]
            current_tsubs = list(itertools.combinations(current_block, 3))
            
            # Find t-subsets that would become uncovered if we remove this block
            critical_tsubs = []
            for t_sub in current_tsubs:
                idx = t_subset_to_idx[t_sub]
                if coverage[idx] == 1:
                    critical_tsubs.append(idx)
            
            if not critical_tsubs:
                continue
            
            # Try to find a replacement block that covers all critical t-subsets
            # and potentially more
            best_replacement = None
            best_gain = 0
            
            # Sample some candidate blocks for efficiency
            candidates = np.random.choice(len(blocks), size=min(1000, len(blocks)), replace=False)
            for cand_idx in candidates:
                cand_block = blocks[cand_idx]
                cand_tsubs = set(itertools.combinations(cand_block, 3))
                
                # Check if candidate covers all critical t-subsets
                critical_set = set(tuple(t_subsets[idx]) for idx in critical_tsubs)
                if not critical_set.issubset(cand_tsubs):
                    continue
                
                # Calculate how many additional t-subsets this would cover
                # that are currently covered only once (would become redundant)
                extra = 0
                for t_sub in cand_tsubs:
                    idx = t_subset_to_idx[t_sub]
                    if coverage[idx] == 1:
                        extra += 1
                
                if extra > best_gain:
                    best_gain = extra
                    best_replacement = cand_idx
            
            if best_replacement is not None and best_replacement >= 0:
                # Perform the swap
                old_block = selected_blocks[i]
                new_block = blocks[best_replacement]
                
                # Update coverage
                for t_sub in itertools.combinations(old_block, 3):
                    coverage[t_subset_to_idx[t_sub]] -= 1
                for t_sub in itertools.combinations(new_block, 3):
                    coverage[t_subset_to_idx[t_sub]] += 1
                
                selected_blocks[i] = list(new_block)
                improved = True
    
    return selected_blocks
