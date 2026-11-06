import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(12)
    
    # 1. Enumerate all 5-subsets and assign indices
    all_5_subsets = list(combinations(points, 5))
    n_5 = len(all_5_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 2. Enumerate all 6-subsets (blocks)
    all_blocks = list(combinations(points, 6))
    
    # 3. For each block, compute its mask of 5-subsets
    block_masks = []
    blocks_covering = [[] for _ in range(n_5)]
    
    for b in all_blocks:
        mask = 0
        for s in combinations(b, 5):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
            blocks_covering[idx].append(len(block_masks))
        block_masks.append(mask)
        
    # 4. Solve Exact Cover
    # Symmetry breaking:
    # The first block is fixed to (0, 1, 2, 3, 4, 5).
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_blocks.index(first_block)
    
    # Initialize
    initial_uncovered = (1 << n_5) - 1
    initial_uncovered &= ~block_masks[first_block_idx]
    
    selected_blocks = [first_block]
    
    # Maintain a list of uncovered subset indices for faster MRV
    # Initially, all subsets except those in the first block
    uncovered_indices = [i for i in range(n_5) if not (initial_uncovered >> i) & 1]
    # Wait, initial_uncovered has 1s for UNCOVERED.
    # So uncovered_indices should be indices where bit is 1.
    uncovered_indices = [i for i in range(n_5) if (initial_uncovered >> i) & 1]
    
    def solve(uncovered_mask, uncovered_indices, selected_count):
        if time.time() > deadline:
            return None
            
        if not uncovered_indices:
            return selected_blocks[:]
            
        # Find the 5-subset with minimum remaining values (MRV)
        min_count = float('inf')
        best_subset_idx = -1
        best_candidates = []
        
        for idx in uncovered_indices:
            count = 0
            candidates = []
            for b_idx in blocks_covering[idx]:
                # Check if block is compatible: it must only cover uncovered subsets
                # i.e., block_mask & ~uncovered_mask == 0
                if block_masks[b_idx] & ~uncovered_mask == 0:
                    count += 1
                    candidates.append(b_idx)
                    
            if count < min_count:
                min_count = count
                best_subset_idx = idx
                best_candidates = candidates
                
            if min_count == 0:
                return None # Dead end
                
            if min_count == 1:
                break # Can't get better than 1 (unless 0)
                
        # Try each candidate for the best_subset_idx
        for b_idx in best_candidates:
            mask = block_masks[b_idx]
            new_uncovered = uncovered_mask & ~mask
            
            # Update uncovered_indices
            # Remove indices covered by mask
            # We can create a new list or modify in place. 
            # Given the small size, creating a new list is fine.
            new_uncovered_indices = [i for i in uncovered_indices if not (mask >> i) & 1]
            
            selected_blocks.append(all_blocks[b_idx])
            
            result = solve(new_uncovered, new_uncovered_indices, selected_count + 1)
            if result is not None:
                return result
                
            selected_blocks.pop()
            
        return None

    result = solve(initial_uncovered, uncovered_indices, 1)
    
    if result is None:
        # Fallback
        return selected_blocks[:]

    return result
