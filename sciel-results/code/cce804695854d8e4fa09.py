import numpy as np
import itertools
import random

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    n_subsets = len(all_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute bitsets for each block
    # Use numpy array of uint64 for speed, or Python ints.
    # n_subsets = 3876. 
    # Let's use Python ints for simplicity and speed on small sizes, 
    # but numpy is better for the "max gain" step.
    
    # Let's use numpy for the main loop.
    # Shape: (n_blocks, n_words)
    n_words = (n_subsets + 63) // 64
    block_bits = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    for i, b in enumerate(all_blocks):
        for sub in itertools.combinations(b, t):
            idx = subset_to_idx[sub]
            word_idx = idx // 64
            bit_idx = idx % 64
            block_bits[i, word_idx] |= (1 << bit_idx)
    
    def popcount_array(arr):
        # arr is 1D array of uint64
        # Use lookup table or builtin if available? 
        # numpy doesn't have built-in popcount for uint64 arrays directly in older versions, 
        # but we can use a trick or just sum bits?
        # Actually, for finding max gain, we need popcount of (block & uncovered).
        # Let's stick to Python ints for the "search" part if numpy popcount is hard, 
        # or implement popcount via lookup.
        
        pass

    # Alternative: Use Python ints for masks.
    # 3876 bits fits in ~60 64-bit words. Python big ints are efficient.
    
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = (1 << n_subsets) - 1
        selected_blocks = []
        
        while uncovered:
            best_block_idx = -1
            best_gain = -1
            best_tie = float('inf')
            
            # Iterate through all blocks
            # This is the bottleneck. 50k blocks.
            # We can prune if we find a block covering all uncovered? Unlikely.
            
            for i in range(n_blocks):
                # Calculate gain
                # mask & uncovered
                # Count bits
                # Python int bit_count is fast.
                gain = (block_masks[i] & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    best_tie = random.random()
                elif gain == best_gain:
                    r = random.random()
                    if r < best_tie:
                        best_tie = r
                        best_block_idx = i
                        
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        # Post-processing: Remove redundant blocks
        # Count coverage
        coverage_count = [0] * n_subsets
        # This is slow. 
        # Better: For each subset, check if it's covered by multiple selected blocks.
        
        # Let's use a set of selected indices.
        sel_set = set(selected_blocks)
        
        # To check redundancy, we need to know if removing a block leaves any subset uncovered.
        # A block is redundant if all its subsets are covered by other selected blocks.
        
        # Efficient redundancy check:
        # For each block B in selected:
        #   Let S_B be subsets in B.
        #   For each s in S_B:
        #      Check if any other block B' in selected (B' != B) covers s.
        # This is O(|Selected| * C(k,t) * |Selected|) which is too slow.
        
        # Better:
        # Maintain a count array.
        counts = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            # Extract bits? No, we have masks.
            # We can iterate through the subsets of the block again.
            for sub in itertools.combinations(all_blocks[b_idx], t):
                idx = subset_to_idx[sub]
                counts[idx] += 1
                
        to_remove = set()
        for b_idx in selected_blocks:
            is_redundant = True
            for sub in itertools.combinations(all_blocks[b_idx], t):
                idx = subset_to_idx[sub]
                if counts[idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.add(b_idx)
                
        # Remove redundant blocks
        final_indices = [b for b in selected_blocks if b not in to_remove]
        
        # Update counts and check again? 
        # The previous code did a loop. Let's do a few passes.
        
        changed = True
        while changed:
            changed = False
            # Recalculate counts
            counts = np.zeros(n_subsets, dtype=np.int32)
            for b_idx in final_indices:
                for sub in itertools.combinations(all_blocks[b_idx], t):
                    idx = subset_to_idx[sub]
                    counts[idx] += 1
            
            to_remove = set()
            for b_idx in final_indices:
                is_redundant = True
                for sub in itertools.combinations(all_blocks[b_idx], t):
                    idx = subset_to_idx[sub]
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.add(b_idx)
            
            if to_remove:
                final_indices = [b for b in final_indices if b not in to_remove]
                changed = True
                
        return [list(all_blocks[i]) for i in final_indices]

    best_len = float('inf')
    best_blocks = []
    
    # Increase restarts due to faster bit ops?
    # 50k blocks * 150 steps = 7.5M bit_counts.
    # Python bit_count is C-speed. 7.5M is ~0.1s?
    # So we can do many restarts.
    
    for seed in range(20):
        blocks = greedy_cover(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    return best_blocks
