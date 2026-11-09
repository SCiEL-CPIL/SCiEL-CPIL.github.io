import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Generate all k-subsets and their covered t-subset indices
    all_k_subs = list(itertools.combinations(range(v), k))
    block_t_lists = []
    for block in all_k_subs:
        t_indices = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_lists.append(t_indices)
        
    num_blocks = len(all_k_subs)
    
    # 2. Initialization: Greedy Set Cover
    # Use bitmasks for fast gain calculation in greedy
    block_masks = []
    for t_indices in block_t_lists:
        mask = 0
        for idx in t_indices:
            mask |= (1 << idx)
        block_masks.append(mask)
            
    uncovered_mask = (1 << num_t_subs) - 1
    solution_indices = []
    
    # Shuffle blocks to add randomness to greedy? 
    # Or just standard greedy. Let's do standard greedy for a solid start.
    # To make it a "new approach", we will follow up with Local Search.
    
    while uncovered_mask:
        best_gain = 0
        best_idx = -1
        
        # Optimization: We can skip blocks that are subsets of already chosen ones? 
        # Hard to track. Just iterate all.
        for i in range(num_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            break # Should not happen
            
        solution_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 3. Local Search: Prune and Swap
    # Maintain counts of coverage
    counts = [0] * num_t_subs
    for idx in solution_indices:
        for t_idx in block_t_lists[idx]:
            counts[t_idx] += 1
            
    # Set of indices in solution for fast lookup
    sol_set = set(solution_indices)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Local Search Loop
    # We try to remove blocks. If stuck, we swap to change the configuration.
    
    while time.time() - start_time < time_limit:
        # Phase 1: Try to remove a random block
        if not solution_indices:
            break
            
        removed_any = False
        # Try a few random removals
        for _ in range(10):
            if not solution_indices:
                break
            i = random.randint(0, len(solution_indices) - 1)
            block_idx = solution_indices[i]
            
            # Tentatively remove
            for t_idx in block_t_lists[block_idx]:
                counts[t_idx] -= 1
                
            # Check validity
            valid = True
            for t_idx in block_t_lists[block_idx]:
                if counts[t_idx] == 0:
                    valid = False
                    break
                    
            if valid:
                # Accept removal
                solution_indices.pop(i)
                sol_set.remove(block_idx)
                removed_any = True
                # Break out to restart loop with new state
                break
            else:
                # Restore
                for t_idx in block_t_lists[block_idx]:
                    counts[t_idx] += 1
                
        if removed_any:
            continue
            
        # Phase 2: If no removal, try a swap to escape local minimum
        if len(solution_indices) < 2:
             break
             
        # Try a few random swaps
        swapped = False
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            # Pick a block to remove (b_out)
            i = random.randint(0, len(solution_indices) - 1)
            b_out = solution_indices[i]
            
            # Pick a block to add (b_in)
            # Random choice from all blocks not in solution
            # To make it faster, we can pick a random index and check if it's in sol_set
            # If it is, retry.
            b_in = random.randint(0, num_blocks - 1)
            if b_in in sol_set:
                continue
                
            # Tentative swap: remove b_out, add b_in
            for t_idx in block_t_lists[b_out]:
                counts[t_idx] -= 1
            for t_idx in block_t_lists[b_in]:
                counts[t_idx] += 1
                
            # Check validity
            valid = True
            # We only need to check if any count is 0.
            # Since we only modified counts for t_idx in b_out and b_in,
            # we only need to check those specific indices?
            # No, if b_out covered a unique t_idx, counts becomes 0.
            # If b_in covers it, counts becomes 1.
            # So we just need to check if min(counts) > 0.
            # Checking all 2024 is fast.
            if min(counts) > 0:
                # Accept swap
                solution_indices[i] = b_in
                sol_set.remove(b_out)
                sol_set.add(b_in)
                swapped = True
                break
            else:
                # Revert
                for t_idx in block_t_lists[b_out]:
                    counts[t_idx] += 1
                for t_idx in block_t_lists[b_in]:
                    counts[t_idx] -= 1
        if swapped:
            continue
        if time.time() - start_time > time_limit:
            break
        # If we couldn't swap or remove, we might be stuck.
        # We can try a more aggressive move: 2-out 1-in? 
        # Or just break if we've been stuck for a while?
        # Let's add a simple 2-out 1-in attempt occasionally.
        
        if random.random() < 0.1:
             # Try 2-out 1-in
             if len(solution_indices) >= 2:
                 i1 = random.randint(0, len(solution_indices) - 1)
                 i2 = random.randint(0, len(solution_indices) - 1)
                 if i1 == i2:
                     continue
                 b_out1 = solution_indices[i1]
                 b_out2 = solution_indices[i2]
                 
                 b_in = random.randint(0, num_blocks - 1)
                 if b_in in sol_set:
                     continue
                     
                 # Tentative
                 for t_idx in block_t_lists[b_out1]:
                     counts[t_idx] -= 1
                 for t_idx in block_t_lists[b_out2]:
                     counts[t_idx] -= 1
                 for t_idx in block_t_lists[b_in]:
                     counts[t_idx] += 1
                     
                 if min(counts) > 0:
                     # Accept
                     # Remove i1, i2 (be careful with indices)
                     if i1 > i2:
                         i1, i2 = i2, i1
                     solution_indices.pop(i2)
                     solution_indices.pop(i1)
                     solution_indices.append(b_in)
                     sol_set.remove(b_out1)
                     sol_set.remove(b_out2)
                     sol_set.add(b_in)
                 else:
                     # Revert
                     for t_idx in block_t_lists[b_out1]:
                         counts[t_idx] += 1
                     for t_idx in block_t_lists[b_out2]:
                         counts[t_idx] += 1
                     for t_idx in block_t_lists[b_in]:
                         counts[t_idx] -= 1

    # Convert indices back to blocks
    return [list(all_k_subs[i]) for i in solution_indices]
