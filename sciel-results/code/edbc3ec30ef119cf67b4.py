import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 145s limit, 5s margin for safety
    
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_subsets = len(t_subsets)
    
    # Precompute masks for all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = {}
    for b in all_blocks:
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= 1 << subset_to_idx[sub]
        block_masks[b] = mask
        
    # Helper to compute mask of a block (tuple or list)
    def get_block_mask(block):
        return block_masks[block]
        
    # Helper to get list of t-subsets in a block
    def get_block_subsets(block):
        return list(itertools.combinations(block, t))

    # --- Phase 1: Fast Greedy Initialization ---
    def greedy_construct():
        uncovered = (1 << n_subsets) - 1
        blocks = []
        counts = [0] * n_subsets
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick the uncovered subset with the lowest index (deterministic for speed)
            idx = (uncovered & -uncovered).bit_length() - 1
            first_subset = t_subsets[idx]
            block = list(first_subset)
            
            # Add k-t more points greedily
            while len(block) < k:
                best_gain = -1
                best_point = -1
                current_block_set = set(block)
                
                for x in range(v):
                    if x in current_block_set:
                        continue
                    
                    # Compute gain: how many new uncovered t-subsets would be covered
                    gain = 0
                    for s in itertools.combinations(block, t-1):
                        sub = tuple(sorted(s + (x,)))
                        idx2 = subset_to_idx[sub]
                        if (uncovered >> idx2) & 1:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and gain > 0 and random.random() < 0.1:
                        best_point = x
                        
                if best_point == -1:
                    break
                block.append(best_point)
                block.sort()
                
            final_block = tuple(block)
            blocks.append(final_block)
            
            mask = block_masks[final_block]
            uncovered &= ~mask
            for sub in itertools.combinations(final_block, t):
                counts[subset_to_idx[sub]] += 1
                
        # Prune redundant blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(blocks)):
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    if counts[subset_to_idx[sub]] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                b = blocks[i]
                blocks.pop(i)
                for sub in itertools.combinations(b, t):
                    counts[subset_to_idx[sub]] -= 1
                changed = True
                
        return blocks, counts

    # --- Phase 2: Simulated Annealing Post-Optimizer ---
    def simulated_annealing(initial_blocks, initial_counts, time_limit):
        # Copy state
        blocks = list(initial_blocks)
        counts = list(initial_counts)
        
        # Verify initial validity
        uncovered = 0
        for i in range(n_subsets):
            if counts[i] == 0:
                uncovered |= (1 << i)
        
        if uncovered != 0:
            # If initial is invalid (shouldn't happen), repair or return
            # For now, just return as is, but we assume greedy is valid
            pass

        best_blocks = list(blocks)
        best_size = len(blocks)
        
        # Temperature schedule
        # Start high, decay exponentially
        T_start = 1.0
        T_end = 0.01
        # Estimate steps: ~1000 steps per second? Let's try 2000 steps/sec
        # We will run until time_limit
        current_time = time.time()
        
        # Precompute list of block indices for random selection
        # We will use random indices directly
        
        while time.time() < deadline - 5: # Safety margin
            T = T_start * (T_end / T_start) ** ((time.time() - start_time) / time_limit)
            if T < T_end:
                T = T_end
            
            # Choose a move:
            # 1. Remove a random block (if it doesn't break coverage)
            # 2. Replace a random block with a new random block (if it improves coverage or maintains)
            
            move_type = random.random()
            
            if move_type < 0.5 and len(blocks) > 1:
                # Try to remove a random block
                idx = random.randrange(len(blocks))
                block_to_remove = blocks[idx]
                
                # Check if any t-subset in this block has count 1
                can_remove = True
                for sub in itertools.combinations(block_to_remove, t):
                    if counts[subset_to_idx[sub]] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove it
                    blocks.pop(idx)
                    for sub in itertools.combinations(block_to_remove, t):
                        counts[subset_to_idx[sub]] -= 1
                    # This is an improvement (size decreases), accept always
                    if len(blocks) < best_size:
                        best_size = len(blocks)
                        best_blocks = list(blocks)
                else:
                    # Cannot remove, try a different move or skip
                    pass
            else:
                # Try to replace a block
                # Pick a random existing block
                idx = random.randrange(len(blocks))
                old_block = blocks[idx]
                
                # Generate a new random block
                new_block = tuple(sorted(random.sample(range(v), k)))
                
                # Calculate change in counts
                # This is O(k*t) which is small
                # But we need to check if it improves "coverage quality"
                # Metric: Number of uncovered subsets (should be 0) or sum of 1/count?
                # Since we want to minimize size, we only accept if size decreases (handled above)
                # OR if size stays same but we enable future removals?
                # Actually, SA usually optimizes a cost function.
                # Cost = size + penalty * (uncovered_subsets)
                # Here uncovered is always 0. So cost = size.
                # Replacing a block keeps size same. It only helps if it makes another block removable.
                # This is hard to evaluate locally.
                
                # Alternative: Randomly swap a block with a "better" one?
                # Or just random walk to escape local minima?
                # Let's try: Replace a block with a random one that covers at least as many "unique" subsets?
                
                # Let's try a simpler heuristic:
                # If we replace a block, we might uncover some subsets.
                # We only accept if the new block covers ALL subsets that the old block uniquely covered.
                
                old_unique = 0
                for sub in itertools.combinations(old_block, t):
                    if counts[subset_to_idx[sub]] == 1:
                        old_unique |= (1 << subset_to_idx[sub])
                        
                if old_unique == 0:
                    # Old block is redundant, we should have removed it.
                    # This implies we missed a removal. Force removal?
                    # Let's just remove it.
                    blocks.pop(idx)
                    for sub in itertools.combinations(old_block, t):
                        counts[subset_to_idx[sub]] -= 1
                    if len(blocks) < best_size:
                        best_size = len(blocks)
                        best_blocks = list(blocks)
                else:
                    # Check if new block covers all old_unique subsets
                    new_mask = block_masks[new_block]
                    if (old_unique & ~new_mask) == 0:
                        # New block covers all unique subsets of old block
                        # Accept the replacement
                        # Update counts
                        for sub in itertools.combinations(old_block, t):
                            counts[subset_to_idx[sub]] -= 1
                        for sub in itertools.combinations(new_block, t):
                            counts[subset_to_idx[sub]] += 1
                        blocks[idx] = new_block
                        # Size unchanged, but maybe enables future removals
                    else:
                        # Reject
                        pass

        return best_blocks

    # Run Greedy
    initial_blocks, initial_counts = greedy_construct()
    
    if time.time() > deadline - 10:
        return [list(b) for b in initial_blocks]
        
    # Run SA
    sa_time_limit = min(10, deadline - time.time() - 5)
    if sa_time_limit > 0:
        final_blocks = simulated_annealing(initial_blocks, initial_counts, sa_time_limit)
    else:
        final_blocks = initial_blocks
        
    return [list(b) for b in final_blocks]
