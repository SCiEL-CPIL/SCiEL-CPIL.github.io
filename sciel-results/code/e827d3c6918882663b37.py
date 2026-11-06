import time
from itertools import combinations

# Global cache for precomputation
_cache = {}

def _get_precomputed():
    if 'data' not in _cache:
        points = range(12)
        all_5subsets = list(combinations(points, 5))
        all_6subsets = list(combinations(points, 6))
        
        n_5 = len(all_5subsets) # 792
        n_6 = len(all_6subsets) # 924
        
        # Map 5-subset to index
        idx_5 = {s: i for i, s in enumerate(all_5subsets)}
        
        # For each 6-subset, store the bitmask of 5-subsets it covers
        block_masks = [0] * n_6
        for i, b in enumerate(all_6subsets):
            mask = 0
            for s in combinations(b, 5):
                mask |= (1 << idx_5[s])
            block_masks[i] = mask
            
        # For each 5-subset, list the indices of 6-subsets that contain it
        subsets_of_5 = [[] for _ in range(n_5)]
        for i, b in enumerate(all_6subsets):
            for s in combinations(b, 5):
                subsets_of_5[idx_5[s]].append(i)
                
        _cache['data'] = (all_5subsets, all_6subsets, n_5, n_6, idx_5, block_masks, subsets_of_5)
    return _cache['data']

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    (all_5subsets, all_6subsets, n_5, n_6, idx_5, block_masks, subsets_of_5) = _get_precomputed()
    
    # Symmetry breaking: Fix the first block
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6subsets.index(first_block)
    
    initial_mask = block_masks[first_block_idx]
    current_blocks = [list(first_block)]
    
    def backtrack(covered_mask, blocks_list, depth):
        if time.time() > deadline:
            return None
        
        if len(blocks_list) == 132:
            return blocks_list
        
        best_5_idx = -1
        min_available = 8
        
        for i in range(n_5):
            if (covered_mask >> i) & 1:
                continue
            
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
            if len(blocks_list) == 132:
                return blocks_list
            return None
        
        for b_idx in subsets_of_5[best_5_idx]:
            mask = block_masks[b_idx]
            if (covered_mask & mask) == 0:
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
        return current_blocks
