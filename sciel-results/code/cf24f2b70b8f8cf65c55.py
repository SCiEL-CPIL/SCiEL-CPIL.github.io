import itertools
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a greedy set cover approach 
    with precomputed block coverage bitmasks, followed by local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    # Generate all k-blocks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # 3. Greedy Set Cover
    # Current uncovered mask: all bits set to 1
    current_uncovered = (1 << num_t_subsets) - 1
    
    chosen_blocks = []
    used_block_indices = set()
    
    # Helper function for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    while current_uncovered != 0:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        # Find the block that covers the most currently uncovered t-subsets
        for i in range(len(block_masks)):
            if i in used_block_indices:
                continue
                
            intersection = block_masks[i] & current_uncovered
            gain = count_bits(intersection)
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            break
            
        chosen_blocks.append(list(all_blocks[best_block_idx]))
        used_block_indices.add(best_block_idx)
        
        current_uncovered &= ~block_masks[best_block_idx]
        
    # 4. Local Search Refinement: Try to remove redundant blocks
    # A block is redundant if removing it doesn't uncover any t-subsets
    # We iterate through blocks and try to remove them if they're redundant
    
    if time.time() < deadline:
        # Compute coverage count for each t-subset
        coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
        for block_idx in used_block_indices:
            # Use numpy to add 1 to positions where mask has bits set
            mask = block_masks[block_idx]
            # Convert mask to array of indices
            if mask != 0:
                # Get indices of set bits
                idxs = np.array([i for i in range(num_t_subsets) if (mask >> i) & 1], dtype=np.int32)
                if len(idxs) > 0:
                    coverage_count[idxs] += 1
        
        # Try to remove blocks that are redundant
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            for block_idx in list(used_block_indices):
                if time.time() > deadline:
                    break
                
                mask = block_masks[block_idx]
                # Check if any t-subset has coverage count of 1 and is covered by this block
                if mask != 0:
                    idxs = np.array([i for i in range(num_t_subsets) if (mask >> i) & 1], dtype=np.int32)
                    if len(idxs) > 0:
                        # If all t-subsets covered by this block have coverage > 1, we can remove it
                        if np.all(coverage_count[idxs] > 1):
                            # Remove this block
                            used_block_indices.discard(block_idx)
                            coverage_count[idxs] -= 1
                            removed_any = True
                            # Remove from chosen_blocks
                            chosen_blocks = [b for b in chosen_blocks if all_blocks[block_idx] != b]
        
        # Reconstruct chosen_blocks from used_block_indices to ensure consistency
        chosen_blocks = [list(all_blocks[i]) for i in used_block_indices]
    
    return chosen_blocks
