import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s buffer for safety
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks) and precompute masks
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # Pre-compute bitmask for each k-subset
    block_masks = [0] * num_k_subsets
    block_lists = [list(block) for block in k_subsets]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # For each t-subset, precompute which blocks cover it
    # This helps with "hardest subset" heuristic
    t_subset_block_counts = [0] * num_t_subsets
    for mask in block_masks:
        # Count bits and add to relevant t-subsets
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            t_subset_block_counts[idx] += 1
            m ^= lsb
    
    # Greedy with hardest-subset tie-breaking
    def greedy_greedy(use_hardest_tiebreak=True):
        uncovered = (1 << num_t_subsets) - 1
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            max_gain = 0
            best_block_idx = -1
            best_mask = 0
            best_hardest_score = -1  # For tie-breaking: min count of blocks covering the hardest uncovered subset
            
            for i in range(num_k_subsets):
                mask = block_masks[i]
                gain = (uncovered & mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
                    best_mask = mask
                    if use_hardest_tiebreak:
                        # Find the "hardest" uncovered subset in this block
                        # (the one with fewest total covering blocks)
                        m = uncovered & mask
                        min_count = float('inf')
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            if t_subset_block_counts[idx] < min_count:
                                min_count = t_subset_block_counts[idx]
                            m ^= lsb
                        best_hardest_score = min_count
                elif gain == max_gain and gain > 0 and use_hardest_tiebreak:
                    # Tie-break: prefer block whose hardest uncovered subset has fewer options
                    m = uncovered & mask
                    min_count = float('inf')
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        if t_subset_block_counts[idx] < min_count:
                            min_count = t_subset_block_counts[idx]
                        m ^= lsb
                    if min_count < best_hardest_score:
                        best_hardest_score = min_count
                        best_block_idx = i
                        best_mask = mask
            
            if best_block_idx == -1:
                break
                
            blocks.append(block_lists[best_block_idx])
            uncovered &= ~best_mask
            
        return blocks
    
    # Local search: remove redundant blocks
    def local_search(blocks):
        if not blocks:
            return blocks
        
        n = len(blocks)
        # For each block, compute what it uniquely covers
        # A block is redundant if removing it doesn't uncover any t-subset
        
        # Build coverage count: for each t-subset, how many blocks cover it
        coverage_count = [0] * num_t_subsets
        block_to_indices = {}
        
        for i, block in enumerate(blocks):
            mask = 0
            for t_sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[t_sub]
                mask |= (1 << idx)
                coverage_count[idx] += 1
            block_to_indices[i] = mask
        
        # Try to remove blocks one by one
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                # Check if block i is redundant
                mask = block_to_indices[i]
                m = mask
                redundant = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if coverage_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    # Remove block i
                    mask = block_to_indices[i]
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m ^= lsb
                    blocks.pop(i)
                    # Re-index remaining blocks
                    for j in range(len(blocks)):
                        pass  # block_to_indices keys are indices, need to rebuild
                    # Rebuild block_to_indices
                    block_to_indices = {}
                    for j, block in enumerate(blocks):
                        mask = 0
                        for t_sub in itertools.combinations(block, t):
                            idx = t_subset_to_idx[t_sub]
                            mask |= (1 << idx)
                        block_to_indices[j] = mask
                    changed = True
                    break
        
        return blocks
    
    # Run greedy with hardest-subset tie-breaking
    blocks1 = greedy_greedy(use_hardest_tiebreak=True)
    
    # Run local search to remove redundant blocks
    blocks1 = local_search(blocks1)
    
    # If time remains, try a second greedy with different tie-breaking
    if time.time() < deadline - 30:
        # Second greedy: standard (no hardest tie-break)
        blocks2 = greedy_greedy(use_hardest_tiebreak=False)
        blocks2 = local_search(blocks2)
        
        # Keep the better solution
        if len(blocks2) < len(blocks1):
            blocks1 = blocks2
    
    return blocks1
