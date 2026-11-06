import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Precomputation
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    all_6subsets = list(combinations(points, 6))
    
    n_5 = len(all_5subsets) # 792
    n_6 = len(all_6subsets) # 924
    
    # Map 5-subset to index
    idx_5 = {s: i for i, s in enumerate(all_5subsets)}
    
    # For each 6-subset, store the bitmask of 5-subsets it covers
    # 792 bits fits in a Python int (arbitrary precision)
    block_masks = [0] * n_6
    for i, b in enumerate(all_6subsets):
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << idx_5[s])
        block_masks[i] = mask
        
    # For each 5-subset, list the indices of 6-subsets that contain it
    # Each 5-subset is in 7 6-subsets
    subsets_of_5 = [[] for _ in range(n_5)]
    for i, b in enumerate(all_6subsets):
        for s in combinations(b, 5):
            subsets_of_5[idx_5[s]].append(i)
            
    # Symmetry breaking: Fix the first block
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6subsets.index(first_block)
    
    initial_mask = block_masks[first_block_idx]
    current_blocks = [list(first_block)]
    
    # Backtracking state
    # We need to find 131 more blocks
    # To speed up, we can also fix the second block?
    # The stabilizer of {0,1,2,3,4,5} in M12 is small.
    # Let's just run the search.
    
    def backtrack(covered_mask, blocks_list, depth):
        if time.time() > deadline:
            return None
        
        if len(blocks_list) == 132:
            return blocks_list
        
        # Find an uncovered 5-subset
        # Iterate through all 792 5-subsets? 
        # Optimization: Keep a list of uncovered 5-subsets?
        # Or just check the mask.
        # Checking 792 bits per step is fast.
        
        # Find the first uncovered 5-subset
        # To be more efficient, we can pick the 5-subset with the fewest available blocks.
        # But since all 5-subsets have exactly 7 candidate blocks, and some are covered,
        # the number of available blocks varies.
        
        best_5_idx = -1
        min_available = 8 # Max is 7, so 8 means "none found yet"
        
        # Heuristic: Pick the 5-subset with the minimum number of available (non-conflicting) 6-subsets
        for i in range(n_5):
            if (covered_mask >> i) & 1:
                continue # Already covered
            
            # Count how many of the 7 blocks containing this 5-subset are still valid
            valid_blocks = 0
            for b_idx in subsets_of_5[i]:
                if (covered_mask & block_masks[b_idx]) == 0:
                    valid_blocks += 1
            
            if valid_blocks < min_available:
                min_available = valid_blocks
                best_5_idx = i
                if min_available == 0:
                    return None # Dead end
                if min_available == 1:
                    break # Can't get better than 1 (unless 0)
        
        if best_5_idx == -1:
            # All 5-subsets covered? Should only happen at depth 132
            if len(blocks_list) == 132:
                return blocks_list
            return None # Should not happen if logic is correct, but just in case
        
        # Try all valid blocks containing best_5_idx
        for b_idx in subsets_of_5[best_5_idx]:
            mask = block_masks[b_idx]
            if (covered_mask & mask) == 0:
                # Try this block
                new_mask = covered_mask | mask
                new_blocks = blocks_list + [list(all_6subsets[b_idx])]
                
                result = backtrack(new_mask, new_blocks, depth + 1)
                if result is not None:
                    return result
                    
        return None

    result = backtrack(initial_mask, current_blocks, 1)
    
    if result:
        return result
    else:
        # Fallback: return the hardcoded known solution if search fails or times out
        # But search should succeed.
        # For safety, if it fails, return the initial block + greedy fill?
        # No, just return the best we can. But it should find 132.
        return current_blocks
