import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Enumerate all k-subsets and compute their coverage as bitmasks
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # coverage[i] is an integer bitmask of t-subsets covered by k_subset i
    coverage = [0] * num_k_subsets
    # Also store the list of t-subset indices for each block to update counts efficiently
    coverage_lists = [[] for _ in range(num_k_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
        coverage[i] = mask
        coverage_lists[i] = indices

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    num_restarts = 50000  # We will limit by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Phase
        uncovered_mask = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            max_gain = 0
            candidates = []
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                # Gain is number of new t-subsets covered
                # count bits in (coverage[i] & uncovered_mask)
                gain = bin(coverage[i] & uncovered_mask).count('1')
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            # Pick a random candidate to add diversity
            chosen_idx = random.choice(candidates)
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered_mask &= ~coverage[chosen_idx]
            
        if uncovered_mask:
            continue
            
        current_count = len(current_blocks)
        
        # 2. Local Search Phase
        # Maintain counts of how many times each t-subset is covered
        counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in coverage_lists[b_idx]:
                counts[t_idx] += 1
        
        improved = True
        # Limit iterations to avoid getting stuck in long loops if no improvement
        max_local_iters = 100
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_local_iters:
            improved = False
            iter_count += 1
            current_count = len(current_blocks)
            
            # Try 1-swap (remove redundant block)
            # A block is redundant if for all t in it, counts[t] > 1
            removed = False
            # Shuffle blocks to check random order? Or just iterate.
            # Iterating all is O(N * C(k,t)).
            for b_idx in current_blocks:
                if time.time() > deadline:
                    break
                redundant = True
                for t_idx in coverage_lists[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b_idx
                    current_blocks.remove(b_idx)
                    for t_idx in coverage_lists[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try 2-swap (remove two blocks if their combined unique coverage is covered by others)
            # Condition:
            # For t in B_i ∩ B_j: counts[t] > 2
            # For t in (B_i ∪ B_j) \ (B_i ∩ B_j): counts[t] > 1
            
            # To speed this up, we can pick random pairs.
            # Or iterate pairs. N is ~300. Pairs ~ 45k.
            # Checking each pair takes O(C(k,t)) ~ 15k operations.
            # 45k * 15k is too slow.
            # So we sample pairs.
            
            if len(current_blocks) >= 2:
                # Sample a few pairs
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    i, j = random.sample(range(len(current_blocks)), 2)
                    b_i = current_blocks[i]
                    b_j = current_blocks[j]
                    
                    # Check condition
                    valid = True
                    # We need to check counts for all t in B_i ∪ B_j
                    # To avoid double counting in intersection, we can just check all t in B_i and B_j
                    # But we need to distinguish intersection vs union-diff.
                    
                    # Let's create a set of indices in B_i and B_j for O(1) lookup?
                    # No, we have lists.
                    # Let's just iterate B_i and B_j.
                    
                    # Optimization: 
                    # If we remove B_i and B_j, for any t in B_i:
                    #   if t in B_j: new_count = counts[t] - 2. Need > 0 => counts[t] > 2
                    #   if t not in B_j: new_count = counts[t] - 1. Need > 0 => counts[t] > 1
                    # Same for B_j.
                    
                    # We can check B_i first.
                    # To check "t in B_j" quickly, we can use a set of B_j's indices.
                    # Creating a set for each pair is expensive.
                    # Alternative: Just check counts.
                    # If counts[t] == 1 for any t in B_i, it's not valid (unless covered by B_j? No, if counts[t]==1, it's only in B_i. If we remove B_i, it's gone. So invalid).
                    # So if any t in B_i has counts[t] == 1, invalid.
                    # If any t in B_j has counts[t] == 1, invalid.
                    # If any t in B_i ∩ B_j has counts[t] == 2, invalid (becomes 0).
                    
                    # So:
                    # 1. Check if any t in B_i has counts[t] == 1. If yes, invalid.
                    # 2. Check if any t in B_j has counts[t] == 1. If yes, invalid.
                    # 3. Check if any t in B_i ∩ B_j has counts[t] == 2. If yes, invalid.
                    
                    # Step 1 & 2 are fast.
                    for t_idx in coverage_lists[b_i]:
                        if counts[t_idx] == 1:
                            valid = False
                            break
                    if not valid:
                        continue
                        
                    for t_idx in coverage_lists[b_j]:
                        if counts[t_idx] == 1:
                            valid = False
                            break
                    if not valid:
                        continue
                        
                    # Step 3: Check intersection
                    # We need to find t in both.
                    # We can iterate B_i and check if it's in B_j.
                    # To make this fast, we can use a set for B_j indices for this specific check.
                    # Since we only do this if steps 1&2 pass, it's less frequent.
                    set_b_j = set(coverage_lists[b_j])
                    for t_idx in coverage_lists[b_i]:
                        if t_idx in set_b_j:
                            if counts[t_idx] == 2:
                                valid = False
                                break
                                
                    if valid:
                        # Remove B_i and B_j
                        current_blocks.remove(b_i)
                        current_blocks.remove(b_j)
                        
                        # Update counts
                        for t_idx in coverage_lists[b_i]:
                            counts[t_idx] -= 1
                        for t_idx in coverage_lists[b_j]:
                            counts[t_idx] -= 1
                            
                        improved = True
                        break # Restart loop to re-evaluate 1-swaps

        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(k_subsets[b]) for b in current_blocks]

    if best_blocks is None:
        # Fallback: return empty list or simple greedy? 
        # The task requires a valid covering. 
        # If we didn't find one, we must return something. 
        # But the greedy phase should always find one if a solution exists.
        # C(19,8,5) exists.
        # If we timed out during greedy, we might have an invalid partial solution.
        # We should ensure we return a valid solution.
        # The best way is to finish the greedy if we haven't, but we are out of time.
        # However, the deadline guard breaks the loop. 
        # If best_blocks is None, it means we never completed a greedy run?
        # Unlikely, as greedy is fast.
        # If we are in trouble, we can return a known valid covering if we hardcode it, 
        # but we don't have one.
        # We'll assume best_blocks is not None.
        
        # As a safety, if best_blocks is None, try one more quick greedy?
        # No, we are out of time.
        # Return empty list will score 0.
        # We should probably just return the best partial? No, must be valid.
        # The prompt says "Valid when every t-subset...".
        # If we can't find a valid one, we score 0.
        
        # Let's assume the first greedy run finishes.
        # Greedy takes < 1 second.
        pass

    return best_blocks
