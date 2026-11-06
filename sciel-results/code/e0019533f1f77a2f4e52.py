import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 10 # Safe margin, search is fast
    
    # Precomputation
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    all_6subsets = list(combinations(points, 6))
    
    n_5 = len(all_5subsets) # 792
    n_6 = len(all_6subsets) # 924
    
    # Map 5-subset to index
    idx_5 = {s: i for i, s in enumerate(all_5subsets)}
    
    # For each 6-subset, store the bitmask of 5-subsets it covers
    block_masks = [0] * n_6
    # Also store the list of 5-subset indices covered by each block for quick removal
    block_5_indices = [None] * n_6
    for i, b in enumerate(all_6subsets):
        mask = 0
        indices = []
        for s in combinations(b, 5):
            idx = idx_5[s]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_5_indices[i] = indices
        
    # For each 5-subset, list the indices of 6-subsets that contain it
    subsets_of_5 = [[] for _ in range(n_5)]
    for i, b in enumerate(all_6subsets):
        for idx in block_5_indices[i]:
            subsets_of_5[idx].append(i)
            
    # Symmetry breaking: Fix the first block
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6subsets.index(first_block)
    
    initial_mask = block_masks[first_block_idx]
    current_blocks = [list(first_block)]
    
    # List of uncovered 5-subset indices
    # Initially all are uncovered except those in the first block
    uncovered_5s = [i for i in range(n_5) if not ((initial_mask >> i) & 1)]
    
    def backtrack(covered_mask, blocks_list, uncovered_list):
        if time.time() > deadline:
            return None
        
        if len(blocks_list) == 132:
            return blocks_list
        
        if not uncovered_list:
            # All covered, but we haven't reached 132 blocks?
            # This should not happen if 132 is the max and we cover 6 each.
            # 132 * 6 = 792.
            # If len(blocks) < 132 and uncovered is empty, we are stuck.
            return None

        # Heuristic: Pick the 5-subset with the minimum number of available blocks
        best_5_idx = -1
        min_available = 8
        
        # Iterate over uncovered 5-subsets
        # To avoid modifying the list while iterating, we just scan it.
        # We don't need to sort or update the list structure, just find the best candidate.
        for i in uncovered_list:
            # Count valid blocks
            valid_blocks = 0
            for b_idx in subsets_of_5[i]:
                if (covered_mask & block_masks[b_idx]) == 0:
                    valid_blocks += 1
            
            if valid_blocks < min_available:
                min_available = valid_blocks
                best_5_idx = i
                if min_available == 0:
                    return None
                if min_available == 1:
                    break
        
        if best_5_idx == -1:
            return None

        # Try all valid blocks containing best_5_idx
        for b_idx in subsets_of_5[best_5_idx]:
            mask = block_masks[b_idx]
            if (covered_mask & mask) == 0:
                # Try this block
                new_mask = covered_mask | mask
                new_blocks = blocks_list + [list(all_6subsets[b_idx])]
                
                # Update uncovered list
                # Remove the 5-subsets covered by this block
                new_uncovered = []
                for idx in uncovered_list:
                    if not ((new_mask >> idx) & 1):
                        new_uncovered.append(idx)
                
                result = backtrack(new_mask, new_blocks, new_uncovered)
                if result is not None:
                    return result
                    
        return None

    result = backtrack(initial_mask, current_blocks, uncovered_5s)
    
    if result:
        return result
    else:
        # Fallback
        return current_blocks
