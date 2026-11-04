import itertools
import random
import time
import numpy as np

# Precomputation for v=19, t=4, k=7
V = 19
K = 7
T = 4

# Map all t-subsets to indices
all_t_subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
num_t_subsets = len(all_t_subsets)

# Precompute for all k-subsets, which t-subsets they cover
all_k_subsets = list(itertools.combinations(range(V), K))
block_to_t_indices = {}
for b in all_k_subsets:
    b_set = set(b)
    covered = []
    for s in all_t_subsets:
        if all(x in b_set for x in s):
            covered.append(subset_to_idx[s])
    block_to_t_indices[b] = covered

# For fast lookup, map block tuple to its t-indices
# We will represent blocks as sorted tuples

def construct(v, k, t):
    # The evaluator passes v, k, t. We assume v=19, k=7, t=4 based on the task description.
    # If parameters differ, we should handle it, but the task is specific.
    if v != 19 or k != 7 or t != 4:
        # Fallback to generic greedy for other parameters
        return _generic_greedy(v, k, t)
        
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # 1. Greedy Construction
    blocks = _greedy_construct()
    
    # 2. Simulated Annealing / Local Search
    best_blocks = blocks[:]
    best_count = len(blocks)
    
    # Initialize coverage counts
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b in blocks:
        for idx in block_to_t_indices[b]:
            cover_count[idx] += 1
            
    # SA Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    no_improve_counter = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            no_improve_counter = 0
            # Reset temperature if we found a new best? Or keep cooling?
            # Keeping cooling is standard, but resetting can help escape.
            # Let's just keep cooling, but if we are stuck, we perturb.
        else:
            no_improve_counter += 1
        
        if no_improve_counter > max_no_improve:
            # Perturbation: Remove random blocks and repair
            blocks = _perturb_and_repair(blocks, cover_count, 3)
            # Rebuild cover_count
            cover_count = np.zeros(num_t_subsets, dtype=np.int32)
            for b in blocks:
                for idx in block_to_t_indices[b]:
                    cover_count[idx] += 1
            T = 0.5 # Reset temperature
            no_improve_counter = 0
            continue
            
        # Generate Move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try Remove
            if len(blocks) > 0:
                idx = random.randint(0, len(blocks) - 1)
                b = blocks[idx]
                # Check if redundant
                is_redundant = True
                for s_idx in block_to_t_indices[b]:
                    if cover_count[s_idx] < 2:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove
                    for s_idx in block_to_t_indices[b]:
                        cover_count[s_idx] -= 1
                    blocks.pop(idx)
                    # Accept
                    no_improve_counter = 0
                else:
                    # Reject
                    pass
        else:
            # Try Swap
            if len(blocks) > 0:
                b_idx = random.randint(0, len(blocks) - 1)
                b = blocks[b_idx]
                # Pick a point to swap
                pos = random.randint(0, k - 1)
                old_point = b[pos]
                # Pick a new point not in b
                candidates = [x for x in range(v) if x not in b]
                if candidates:
                    new_point = random.choice(candidates)
                    # Create new block
                    new_b_list = list(b)
                    new_b_list[pos] = new_point
                    new_b = tuple(sorted(new_b_list))
                    
                    if new_b == b:
                        continue
                        
                    # Check validity of swap
                    # 1. Decrement old block's coverage
                    old_indices = block_to_t_indices[b]
                    for s_idx in old_indices:
                        cover_count[s_idx] -= 1
                    
                    # 2. Check if any uncovered
                    valid = True
                    for s_idx in old_indices:
                        if cover_count[s_idx] < 0:
                            valid = False
                            break
                        # Note: cover_count can be 0 here, that's fine if new block covers it or if it was covered by others.
                        # Wait, we just decremented. If it was 1 and we decremented, it's 0.
                        # We need to check if it's covered by the NEW block or others.
                        # So we must add the new block's coverage first or check carefully.
                        
                    # Better approach:
                    # 1. Decrement old
                    # 2. Increment new
                    # 3. Check if any cover_count < 1
                    # 4. If not, accept. Else revert.
                    
                    # Let's redo the check properly.
                    # We already decremented old.
                    new_indices = block_to_t_indices[new_b]
                    for s_idx in new_indices:
                        cover_count[s_idx] += 1
                        
                    # Now check validity
                    for s_idx in old_indices:
                        if cover_count[s_idx] < 1:
                            valid = False
                            break
                    if valid:
                        for s_idx in new_indices:
                            if cover_count[s_idx] < 1:
                                valid = False
                                break
                                # This is redundant if we just incremented, but the old ones might be the problem.
                                # Actually, if we incremented new, they are >= 1.
                                # The only risk is the old ones that are NOT in the new block.
                    
                    # Simplified Validity Check:
                    # A swap is valid if for every t-subset s:
                    #   (s in old_b and s not in new_b) => covered by other blocks (count > 0 after removing old, before adding new? No.)
                    #   Let's just use the final cover_count.
                    #   If any cover_count < 1, it's invalid.
                    
                    # Re-evaluate:
                    # We decremented old, incremented new.
                    # If any cover_count < 1, it means some subset is not covered by any block.
                    # This can only happen for subsets that were in old_b but not in new_b (since new_b subsets are now >= 1).
                    # So we only need to check subsets in old_indices that are NOT in new_indices.
                    
                    valid = True
                    # Optimization: Only check old_indices
                    for s_idx in old_indices:
                        if cover_count[s_idx] < 1:
                            valid = False
                            break
                            
                    if valid:
                        blocks[b_idx] = new_b
                        no_improve_counter = 0
                    else:
                        # Revert
                        for s_idx in new_indices:
                            cover_count[s_idx] -= 1
                        for s_idx in old_indices:
                            cover_count[s_idx] += 1
        
        # Cooling
        if T > min_T:
            T *= cooling_rate
        else:
            # If T is very low, we might be stuck.
            # Force a perturbation sooner?
            no_improve_counter += 100

    # Return the best blocks found
    # Ensure format is list of lists
    return [list(b) for b in best_blocks]

def _greedy_construct():
    uncovered = set(range(num_t_subsets))
    blocks = []
    # Precompute for each t-subset index, which blocks cover it?
    # Or just iterate through all blocks?
    # There are 50388 blocks. 3876 subsets.
    # For each uncovered subset, find the block that covers the most uncovered subsets.
    
    # To speed up, we can precompute a list of blocks for each t-subset.
    # t_subset_to_blocks = {idx: [block_tuples...]}
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b, indices in block_to_t_indices.items():
        for idx in indices:
            t_subset_to_blocks[idx].append(b)
            
    while uncovered:
        # Pick a random uncovered subset
        s_idx = random.choice(list(uncovered))
        # Find best block covering s_idx
        best_block = None
        max_gain = -1
        for b in t_subset_to_blocks[s_idx]:
            gain = 0
            for idx in block_to_t_indices[b]:
                if idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b
                
        if best_block is None:
            break
            
        blocks.append(best_block)
        for idx in block_to_t_indices[best_block]:
            if idx in uncovered:
                uncovered.remove(idx)
                
    return blocks

def _perturb_and_repair(blocks, cover_count, num_remove):
    # Remove num_remove random blocks
    indices_to_remove = random.sample(range(len(blocks)), min(num_remove, len(blocks)))
    for idx in sorted(indices_to_remove, reverse=True):
        b = blocks[idx]
        for s_idx in block_to_t_indices[b]:
            cover_count[s_idx] -= 1
        blocks.pop(idx)
        
    # Identify uncovered subsets
    uncovered = set()
    for i in range(num_t_subsets):
        if cover_count[i] < 1:
            uncovered.add(i)
            
    # Greedy repair
    while uncovered:
        s_idx = random.choice(list(uncovered))
        best_block = None
        max_gain = -1
        for b in t_subset_to_blocks[s_idx]:
            gain = 0
            for idx in block_to_t_indices[b]:
                if idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b
        if best_block is None:
            break
        blocks.append(best_block)
        for idx in block_to_t_indices[best_block]:
            if idx in uncovered:
                uncovered.remove(idx)
            cover_count[idx] += 1
            
    return blocks

def _generic_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
