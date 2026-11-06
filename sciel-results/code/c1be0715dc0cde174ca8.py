import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 12
    # All 5-subsets
    all_5_subsets = list(combinations(range(n), 5))
    # Map 5-subsets to indices for bitmask representation
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    num_subsets = len(all_5_subsets) # 792
    
    # All 6-subsets (blocks)
    all_blocks = list(combinations(range(n), 6))
    
    # Precompute for each block, the bitmask of 5-subsets it covers
    block_masks = []
    for b in all_blocks:
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << subset_to_idx[s])
        block_masks.append(mask)
        
    # Precompute for each 5-subset index, the list of blocks containing it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            subset_to_blocks[idx].append(i)
            m ^= lsb
            
    # Symmetry Breaking:
    # Fix Block 0: {0, 1, 2, 3, 4, 5}
    block0_idx = 0
    target_block_0 = (0, 1, 2, 3, 4, 5)
    if all_blocks[block0_idx] != target_block_0:
        block0_idx = all_blocks.index(target_block_0)
        
    # Fix Block 1: {0, 1, 2, 3, 6, 7}
    target_block_1 = (0, 1, 2, 3, 6, 7)
    block1_idx = all_blocks.index(target_block_1)
    
    initial_mask = block_masks[block0_idx] | block_masks[block1_idx]
    selected = [block0_idx, block1_idx]
    
    full_mask = (1 << num_subsets) - 1
    
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(current_mask):
        if time.time() > deadline:
            return None
            
        if len(selected) == 132:
            if current_mask == full_mask:
                return selected[:]
            return None
            
        # Find an uncovered 5-subset
        # We pick the first one found for simplicity, but checking all might allow better MRV
        # Given the small size, first-found is likely fine due to symmetry breaking.
        target_subset_idx = -1
        for i in range(num_subsets):
            if not (current_mask & (1 << i)):
                target_subset_idx = i
                break
                
        if target_subset_idx == -1:
            return None
            
        candidates = subset_to_blocks[target_subset_idx]
        
        valid_candidates = []
        for b_idx in candidates:
            if not (block_masks[b_idx] & current_mask):
                valid_candidates.append(b_idx)
                
        if not valid_candidates:
            return None
            
        # Heuristic: Sort candidates by how many new subsets they cover?
        # All cover 6.
        # Maybe sort by index?
        
        for b_idx in valid_candidates:
            new_mask = current_mask | block_masks[b_idx]
            selected.append(b_idx)
            
            result = backtrack(new_mask)
            if result is not None:
                return result
                
            selected.pop()
            
        return None
        
    result = backtrack(initial_mask)
    
    if result is None:
        # Fallback if symmetry breaking assumption fails (should not happen)
        # Try greedy?
        covered, blocks = set(), []
        for b in combinations(range(n), 6):
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks

    blocks = [list(all_blocks[i]) for i in result]
    return blocks
