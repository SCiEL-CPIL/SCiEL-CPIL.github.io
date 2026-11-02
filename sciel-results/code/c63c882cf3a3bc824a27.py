import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Precompute
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Map each block to the indices of t-subsets it covers
    # And a bitmask
    all_blocks = list(itertools.combinations(points, k))
    block_masks = []
    block_t_indices = []
    for b in all_blocks:
        b_set = set(b)
        idxs = []
        mask = 0
        for ts in t_subsets:
            if set(ts).issubset(b_set):
                idx = t_index[ts]
                idxs.append(idx)
                mask |= (1 << idx)
        block_masks.append(mask)
        block_t_indices.append(idxs)
        
    # For fast lookup: which blocks contain a given t-subset?
    # t_containing_blocks[ts_index] = list of block_indices
    t_containing_blocks = [[] for _ in range(num_t)]
    for b_idx, b in enumerate(all_blocks):
        b_set = set(b)
        for ts in t_subsets:
            if set(ts).issubset(b_set):
                t_containing_blocks[t_index[ts]].append(b_idx)
                
    best_blocks = None
    best_size = float('inf')
    
    # 2. Greedy + Refinement Loop
    while time.time() < deadline:
        # Greedy Phase
        cover_count = [0] * num_t
        uncovered_mask = (1 << num_t) - 1
        current_blocks = []
        
        # To improve greedy, we pick a seed t-subset from uncovered ones
        # and choose the best block containing it.
        # We try a few random seeds to escape local minima.
        
        while uncovered_mask:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Pick a few random uncovered t-subsets to consider
            # Get list of uncovered indices
            # Efficient way to get a random uncovered bit:
            # We can maintain a list of uncovered indices, but it's expensive to update.
            # Instead, we can pick a random number in [0, num_t) and check if bit is set.
            # Or, we can just pick the first one?
            # Let's pick 5 random uncovered t-subsets.
            
            candidates_seeds = []
            attempts = 0
            while len(candidates_seeds) < 5 and attempts < 50:
                idx = random.randint(0, num_t - 1)
                if (uncovered_mask >> idx) & 1:
                    candidates_seeds.append(idx)
                attempts += 1
            
            if not candidates_seeds:
                # Fallback: find first uncovered
                # This should not happen if uncovered_mask != 0
                for i in range(num_t):
                    if (uncovered_mask >> i) & 1:
                        candidates_seeds.append(i)
                        break
            
            best_block_idx = -1
            best_gain = -1
            
            for seed_idx in candidates_seeds:
                # Get blocks containing this seed
                for b_idx in t_containing_blocks[seed_idx]:
                    gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
            
            if best_block_idx == -1:
                break # Should not happen if problem is solvable
            
            # Add block
            current_blocks.append(best_block_idx)
            # Update cover counts and uncovered mask
            b_mask = block_masks[best_block_idx]
            # We need to update cover_count for indices where b_mask is 1
            # And remove them from uncovered_mask if count becomes 1
            # Iterating over all bits in b_mask is O(k choose t) = 35
            for idx in block_t_indices[best_block_idx]:
                cover_count[idx] += 1
                if cover_count[idx] == 1:
                    uncovered_mask &= ~(1 << idx)
        
        if time.time() > deadline:
            break
            
        # Refinement Phase: Randomized Deletion
        # Shuffle and try to remove
        random.shuffle(current_blocks)
        # We need to be careful with the order if we remove.
        # We'll iterate through the shuffled list.
        
        # To allow more removals, we can do multiple passes.
        # Pass 1: Shuffle, try remove.
        # Pass 2: Shuffle again, try remove.
        
        for _ in range(2): # 2 passes
            random.shuffle(current_blocks)
            new_blocks = []
            for b_idx in current_blocks:
                # Check if b_idx can be removed
                # It can be removed if for all idx in block_t_indices[b_idx], cover_count[idx] > 1
                can_remove = True
                for idx in block_t_indices[b_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for idx in block_t_indices[b_idx]:
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            # This shouldn't happen if can_remove is true
                            pass
                else:
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
        
        # Check if this is better
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = current_blocks[:]
            
        # If we reached the theoretical lower bound or a known good bound, we could stop,
        # but we don't know the exact best. 151 is known.
        # We just keep searching for better.
        
    # Convert indices back to blocks
    if best_blocks is None:
        # Fallback: return empty or error? Should not happen.
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
