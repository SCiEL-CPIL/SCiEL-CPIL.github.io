import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage as bitmasks
    # block_mask[i] is a bitmask of length num_t_subsets where bit j is 1 if block i covers t_subset j
    # Note: Python integers are arbitrary precision, so this is efficient.
    block_masks = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Precompute for each t-subset index, which blocks cover it (for fast candidate generation in SA)
    # This is the inverse index.
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        mask = block_masks[i]
        # Iterate set bits
        # Since num_t_subsets is small (11628 for 19,5), we can just check all or use bit tricks.
        # For 19,5, C(19,5)=11628. 
        # A faster way: for each t_subset, we know which blocks contain it.
        # But building this by scanning all blocks is O(N_blocks * N_t_subsets) which is ~800M ops. Too slow.
        # We must build it during the block generation loop.
        
    # Let's rebuild t_to_blocks efficiently during block generation
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            t_to_blocks[idx].append(i)
            
    # 2. Initial Solution: Greedy Set Cover
    # We want to minimize the number of blocks.
    # Greedy: pick block covering most uncovered t-subsets.
    
    uncovered_mask = (1 << num_t_subsets) - 1
    current_blocks = []
    # To speed up greedy, we can precompute the "gain" but it changes as we pick blocks.
    # Simple greedy:
    # Keep a list of active blocks (all initially).
    # At each step, find block with max (block_mask & uncovered_mask).bit_count()
    
    # Optimization: Precompute bit_count for all blocks? No, depends on uncovered_mask.
    # We can just iterate. N_blocks=75582. Steps ~300. 22M bit_counts. Python bit_count is fast.
    
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Random tie-breaking to vary initial solution? Or deterministic?
        # Deterministic is fine for start, SA will handle diversity.
        for i in range(num_blocks):
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        current_blocks.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 3. Simulated Annealing
    # State: list of block indices, and a mask of covered t-subsets.
    # Objective: Minimize len(current_blocks) + penalty * (uncovered_count)
    # We want to reach uncovered_count = 0 with min blocks.
    
    # Initialize SA parameters
    T = 1.0 # Temperature
    cooling_rate = 0.995
    min_T = 0.01
    max_iterations = 500000 # Cap iterations, but time is the real limit
    
    best_solution = current_blocks[:]
    best_cost = len(best_solution)
    
    # Current state
    curr_blocks = current_blocks[:]
    curr_mask = 0
    for b in curr_blocks:
        curr_mask |= block_masks[b]
    curr_uncovered = num_t_subsets - curr_mask.bit_count()
    curr_cost = len(curr_blocks) + curr_uncovered * 100 # Heavy penalty for uncovered
    
    best_cost_val = curr_cost
    
    # Helper to calculate cost
    def get_cost(blocks, mask):
        unc = num_t_subsets - mask.bit_count()
        return len(blocks) + unc * 100

    # Precompute block indices for random access
    block_indices = list(range(num_blocks))
    
    while time.time() - start_time < time_limit and T > min_T:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random block
            if len(curr_blocks) == 0:
                continue
            idx_to_remove = random.randint(0, len(curr_blocks) - 1)
            b_idx = curr_blocks[idx_to_remove]
            
            # Calculate new mask
            # We need to know if any t-subset becomes uncovered.
            # new_mask = curr_mask & ~block_masks[b_idx] is WRONG because other blocks might cover it.
            # We need to recalculate the mask from scratch or maintain counts.
            # Recalculating mask from scratch is O(N_blocks * bit_ops). Too slow.
            # We must maintain a count of how many blocks cover each t-subset.
            
            # Let's switch to a count-based representation for the mask to allow O(1) updates?
            # Or just recalculate the "uncovered" part efficiently?
            # Actually, for v=19, k=8, t=5, the number of blocks is small enough that we can
            # just recalculate the mask if we do it smartly? No, 300 blocks * 56 subsets = 16800 ops.
            # Doing this every iteration (500k) is 8 billion ops. Too slow.
            
            # We MUST maintain a count array: count[j] = number of blocks covering t_subset j.
            # This allows O(1) update for add/remove block.
            
            # Let's refactor to use counts.
            pass # This branch is handled below with the refactored logic

        # --- Refactored Logic for Speed ---
        # We need to maintain `counts` array.
        # Let's restart the SA loop with proper state management.
        
        break # Exit dummy loop to implement proper SA below
        
    # --- Proper SA Implementation ---
    
    # State:
    # curr_blocks: list of block indices
    # counts: array of int, counts[j] = num blocks covering t_subset j
    # uncovered_count: int
    # cost: len(curr_blocks) + uncovered_count * penalty
    
    curr_blocks = best_solution[:]
    # Precompute block_to_t_indices
    block_to_t_indices = []
    for i, block in enumerate(all_blocks):
        indices = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_t_indices.append(indices)
        
    # Initialize counts
    counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b in curr_blocks:
        for idx in block_to_t_indices[b]:
            counts[idx] += 1
            
    uncovered_count = np.sum(counts == 0)
    curr_cost = len(curr_blocks) + uncovered_count * 100
    best_cost_val = curr_cost
    best_blocks = curr_blocks[:]
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    penalty = 100
    
    while time.time() - start_time < time_limit and T > min_T:
        # Pick a move
        r = random.random()
        
        if r < 0.4:
            # Move: Remove a random block
            if len(curr_blocks) == 0:
                continue
            pos = random.randint(0, len(curr_blocks) - 1)
            b_idx = curr_blocks[pos]
            
            # Delta cost:
            # Removing b_idx reduces len by 1.
            # It may increase uncovered_count.
            # A t-subset becomes uncovered if counts[idx] == 1 before removal.
            new_uncovered_inc = 0
            for idx in block_to_t_indices[b_idx]:
                if counts[idx] == 1:
                    new_uncovered_inc += 1
            
            delta_len = -1
            delta_unc = new_uncovered_inc
            delta_cost = delta_len + delta_unc * penalty
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept move
                curr_blocks.pop(pos)
                for idx in block_to_t_indices[b_idx]:
                    counts[idx] -= 1
                uncovered_count += new_uncovered_inc
                curr_cost += delta_cost
                
                if curr_cost < best_cost_val:
                    best_cost_val = curr_cost
                    best_blocks = curr_blocks[:]
                    
        elif r < 0.8:
            # Move: Add a random block
            # To be effective, we should add a block that covers some uncovered subsets.
            # Pick a random uncovered subset, then a random block covering it.
            if uncovered_count > 0:
                # Find an uncovered index
                # Efficient way: keep a list of uncovered indices? 
                # Or just pick random indices until we find an uncovered one.
                # Since we start with a good greedy solution, uncovered_count is likely 0.
                # If uncovered_count is 0, adding a block increases cost (len+1).
                # So we should only add if it allows removing more later?
                # In SA, we accept worse moves to escape.
                
                # If uncovered is 0, adding a block is bad.
                # But maybe we can swap?
                # Let's implement a Swap move instead of pure Add.
                
                # Swap: Remove one, Add one.
                if len(curr_blocks) > 0:
                    pos = random.randint(0, len(curr_blocks) - 1)
                    old_b = curr_blocks[pos]
                    
                    # Pick a new block.
                    # Strategy: Pick a block that covers many of the subsets covered by old_b,
                    # or covers uncovered ones.
                    # Simple: Pick a random block from all_blocks.
                    # Better: Pick a block that shares at least one t-subset with old_b?
                    # Let's just pick a random block from all_blocks for now.
                    new_b = random.randint(0, num_blocks - 1)
                    
                    if new_b == old_b:
                        continue
                        
                    # Calculate delta
                    # Remove old_b
                    # Add new_b
                    
                    # We need to compute the change in counts and uncovered.
                    # This is complex to do in O(1).
                    # We can do it in O(C(k,t)) which is 56.
                    
                    # Temporarily simulate
                    # For each idx in old_b:
                    #   if counts[idx] == 1:
                    #      if new_b does NOT cover idx: uncovered_inc += 1
                    #      else: covered by new_b, so no change in uncovered status?
                    #      Wait, if counts[idx]==1 and new_b covers it, then after remove old, count=0, then add new, count=1. Net change 0.
                    #      If counts[idx]>1, no change in uncovered status.
                    
                    # For each idx in new_b:
                    #   if counts[idx] == 0:
                    #      if old_b does NOT cover idx: uncovered_dec += 1
                    #      else: (impossible if counts[idx]==0 and old_b covers it? No, if old_b covers it, counts[idx] was at least 1 before removal? 
                    #             Wait, we are doing a swap. 
                    #             State before: counts.
                    #             State after: counts - old_b + new_b.
                    
                    # Let's just recalculate the relevant parts.
                    # Create a set of indices touched by old_b or new_b.
                    # touched = set(block_to_t_indices[old_b]) | set(block_to_t_indices[new_b])
                    # For each idx in touched:
                    #   old_count = counts[idx]
                    #   new_count = old_count
                    #   if idx in block_to_t_indices[old_b]: new_count -= 1
                    #   if idx in block_to_t_indices[new_b]: new_count += 1
                    #   if old_count == 0 and new_count > 0: uncovered_dec += 1
                    #   if old_count > 0 and new_count == 0: uncovered_inc += 1
                    
                    # This is O(112) operations. Very fast.
                    
                    old_indices = block_to_t_indices[old_b]
                    new_indices = block_to_t_indices[new_b]
                    
                    # Use a set for old indices for fast lookup
                    old_set = set(old_indices)
                    new_set = set(new_indices)
                    
                    touched = old_set.union(new_set)
                    
                    unc_inc = 0
                    unc_dec = 0
                    for idx in touched:
                        c = counts[idx]
                        nc = c
                        if idx in old_set:
                            nc -= 1
                        if idx in new_set:
                            nc += 1
                        if c == 0 and nc > 0:
                            unc_dec += 1
                        elif c > 0 and nc == 0:
                            unc_inc += 1
                            
                    delta_len = 0 # Swap keeps length same
                    delta_unc = unc_inc - unc_dec
                    delta_cost = delta_len + delta_unc * penalty
                    
                    if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                        # Accept swap
                        # Update counts
                        for idx in old_indices:
                            counts[idx] -= 1
                        for idx in new_indices:
                            counts[idx] += 1
                        uncovered_count += delta_unc
                        curr_cost += delta_cost
                        curr_blocks[pos] = new_b
                        
                        if curr_cost < best_cost_val:
                            best_cost_val = curr_cost
                            best_blocks = curr_blocks[:]
            else:
                # No uncovered subsets, so adding is bad. 
                # We can try to remove a redundant block (should be handled by r<0.4)
                # Or just do a swap to explore.
                if len(curr_blocks) > 0:
                    pos = random.randint(0, len(curr_blocks) - 1)
                    old_b = curr_blocks[pos]
                    new_b = random.randint(0, num_blocks - 1)
                    if new_b == old_b: continue
                    
                    old_indices = block_to_t_indices[old_b]
                    new_indices = block_to_t_indices[new_b]
                    old_set = set(old_indices)
                    new_set = set(new_indices)
                    touched = old_set.union(new_set)
                    
                    unc_inc = 0
                    unc_dec = 0
                    for idx in touched:
                        c = counts[idx]
                        nc = c
                        if idx in old_set: nc -= 1
                        if idx in new_set: nc += 1
                        if c == 0 and nc > 0: unc_dec += 1
                        elif c > 0 and nc == 0: unc_inc += 1
                        
                    delta_cost = (unc_inc - unc_dec) * penalty
                    if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                        for idx in old_indices: counts[idx] -= 1
                        for idx in new_indices: counts[idx] += 1
                        uncovered_count += (unc_inc - unc_dec)
                        curr_cost += delta_cost
                        curr_blocks[pos] = new_b
                        if curr_cost < best_cost_val:
                            best_cost_val = curr_cost
                            best_blocks = curr_blocks[:]
        else:
            # Move: Add a block (only if it covers uncovered)
            if uncovered_count > 0:
                # Pick random uncovered index
                # We need to find an index where counts[idx]==0
                # We can keep a list of uncovered indices to make this O(1).
                # But updating that list is overhead.
                # For now, sample random indices.
                found = False
                for _ in range(10):
                    idx = random.randint(0, num_t_subsets - 1)
                    if counts[idx] == 0:
                        # Pick a random block covering this idx
                        candidates = t_to_blocks[idx]
                        if candidates:
                            new_b = random.choice(candidates)
                            # Check if new_b is already in curr_blocks
                            if new_b in curr_blocks:
                                continue
                            
                            # Add new_b
                            # Delta: len +1, uncovered - (number of uncovered idx covered by new_b)
                            unc_dec = 0
                            for i in block_to_t_indices[new_b]:
                                if counts[i] == 0:
                                    unc_dec += 1
                                    
                            delta_cost = 1 - unc_dec * penalty
                            
                            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                curr_blocks.append(new_b)
                                for i in block_to_t_indices[new_b]:
                                    counts[i] += 1
                                uncovered_count -= unc_dec
                                curr_cost += delta_cost
                                
                                if curr_cost < best_cost_val:
                                    best_cost_val = curr_cost
                                    best_blocks = curr_blocks[:]
                            found = True
                            break
                            
        T *= cooling_rate
        
    # Final check: Ensure we return a valid covering (uncovered_count == 0)
    # If SA ended with uncovered subsets, we need to fix it.
    # The greedy initial solution was valid. If we ended up invalid, we should return the best valid solution found.
    # But best_blocks might be invalid if best_cost_val included penalty?
    # No, we track best_cost_val which includes penalty.
    # We should track the best VALID solution separately?
    # Or just ensure that if uncovered_count > 0, we don't update best?
    # In the code above, we update best_cost_val regardless.
    # If the final best_blocks has uncovered subsets, it's invalid.
    # We must return a valid covering.
    
    # Check if best_blocks is valid
    # Recalculate mask for best_blocks
    best_mask = 0
    for b in best_blocks:
        best_mask |= block_masks[b]
    if best_mask.bit_count() < num_t_subsets:
        # Invalid. Fall back to the initial greedy solution which was valid.
        # Or try to repair?
        # The greedy solution is valid.
        # Let's just return the greedy solution if SA failed to maintain validity.
        # Actually, the greedy solution is a valid covering.
        # We can just return the best valid solution found.
        # Let's re-evaluate the best valid solution.
        
        # We didn't track "best_valid_cost" separately.
        # Let's just use the initial greedy solution if the SA result is invalid.
        # This is a safe fallback.
        
        # To be better, we can try to repair best_blocks by adding blocks for uncovered subsets.
        # But for now, fallback to greedy.
        
        # Wait, the greedy solution is `current_blocks` from step 2.
        # We stored it in `best_solution` initially.
        # But we modified `curr_blocks`.
        # Let's keep a copy of the initial greedy solution.
        
        # I didn't keep a separate copy of the initial greedy solution in the refactored part.
        # I set `best_solution = current_blocks[:]` before SA.
        # But then I overwrote `curr_blocks` and updated `best_blocks`.
        # If `best_blocks` is invalid, I should return the initial greedy.
        
        # Let's re-run a quick greedy repair on `best_blocks` if it's invalid?
        # Or just return the initial greedy.
        
        # Let's assume the SA is good enough to stay valid or close to valid.
        # If it's invalid, we can just add blocks to cover the missing parts.
        
        # Repair:
        uncovered_idx = [i for i in range(num_t_subsets) if (best_mask >> i) & 1 == 0]
        for idx in uncovered_idx:
            # Find a block covering idx
            cands = t_to_blocks[idx]
            if cands:
                # Pick one not in best_blocks
                for c in cands:
                    if c not in best_blocks:
                        best_blocks.append(c)
                        break
                else:
                    best_blocks.append(cands[0])
                    
    return [list(all_blocks[i]) for i in best_blocks]
