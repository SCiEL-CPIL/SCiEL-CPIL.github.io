"""
Strategy: Iterated Local Search (ILS) with Redundancy Removal and Repair
Instead of building a covering from scratch using greedy addition (which gets stuck
in local minima around 136-150 blocks), we start with a valid covering and then
try to reduce its size.

1. Initialization: Use a fast greedy algorithm to generate a valid covering.
2. Optimization Loop:
   a. Identify redundant blocks (blocks whose removal does not uncover any t-subsets).
   b. Remove redundant blocks.
   c. Attempt "swaps": Replace a block with another block that covers the same
      unique t-subsets (if any) but potentially allows for future redundancy removal,
      or simply try to find a smaller covering by random restarts of the greedy
      process with different random seeds, keeping the best found.
   
   However, a more effective ILS approach for set cover is:
   - Start with a covering.
   - Randomly perturb by removing a random set of blocks (making it invalid).
   - Repair the invalid covering by adding blocks greedily to cover the uncovered t-subsets.
   - Accept the new covering if it is smaller or equal (with some probability for larger ones to escape local minima).
   
   Given the time limit (120s), we will implement a robust version:
   1. Fast Greedy Construction (optimized with numpy) to get a baseline.
   2. Multiple restarts with different random seeds for the greedy construction.
   3. Post-processing: Redundancy removal.
   4. If time permits, simple local search: try to replace blocks to reduce count.
   
   Actually, the "constrained swaps" insight suggests maintaining validity.
   Let's do:
   1. Generate a valid covering C.
   2. While time remains:
      a. Try to remove a random block b from C. If C\{b} is still valid, keep it removed.
      b. If not valid, try to find a block b' not in C such that C - {b} + {b'} is valid 
         and ideally allows for further reductions (hard to predict, so just check validity).
         Actually, if C\{b} is invalid, we can't just swap unless we add b'.
         A better move: Pick a block b in C. Pick a block b' not in C.
         Check if C - {b} + {b'} is valid. If yes, accept if it helps (it doesn't change size, 
         but might enable future removals).
      
   The most effective simple strategy that often beats greedy is **Multiple Random Greedy** 
   followed by **Redundancy Removal**. Greedy is deterministic if we always pick the max gain.
   If we break ties randomly, we get different coverings.
   
   Let's implement:
   1. Precompute all block gains (masks of t-subsets covered).
   2. Run Greedy with random tie-breaking multiple times.
   3. For each resulting covering, perform redundancy removal.
   4. Keep the smallest covering.
   5. If time remains, perform local search: 
      - For each block in the covering, try to remove it. If invalid, try to add a new block 
        that covers the "lost" t-subsets while keeping the block removed? No, that increases size.
      - Actually, just stick to Multiple Random Greedy + Redundancy Removal. It is fast and robust.
"""

import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    # Deadline management: leave 5 seconds for final return and safety
    deadline = time.time() + 115.0
    
    # 1. Precomputation
    # Generate all t-subsets as bitmasks and map to indices
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    
    n_tsub = len(tsub_masks)
    idx_of = {m: i for i, m in enumerate(tsub_masks)}
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # Precompute the t-subset indices covered by each k-subset
    # k_sub_tsub_indices[i] is a list of indices of t-subsets covered by block i
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        subs = []
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.append(idx_of[sm])
        k_sub_tsub_indices.append(subs)
    
    # Create a flat array and offsets for efficient numpy operations
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # Helper function to run one greedy pass with a specific random seed for tie-breaking
    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsub, dtype=bool)
        selected_blocks = []
        
        # Precompute gains? No, gains change as uncovered changes.
        # We need to calculate gain for all blocks in each step.
        # To speed this up, we can use numpy sum over slices.
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            # Calculate gain for all blocks
            # gain[i] = sum(uncovered[all_indices_arr[offsets[i]:offsets[i+1]]])
            gains = np.zeros(n_ksub, dtype=np.int32)
            
            # This loop is the bottleneck. 50388 blocks * 35 t-subsets each.
            # We can try to vectorize this partially or just optimize the python loop.
            # Using numpy sum on slices is reasonably fast.
            for i in range(n_ksub):
                start = offsets_arr[i]
                end = offsets_arr[i+1]
                if start < end:
                    gains[i] = np.sum(uncovered[all_indices_arr[start:end]])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if uncovered.any() is true and we have all blocks
                break
            
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            selected_blocks.append(best_idx)
            
            # Update uncovered
            start = offsets_arr[best_idx]
            end = offsets_arr[best_idx+1]
            uncovered[all_indices_arr[start:end]] = False
            
        return selected_blocks, uncovered
    
    # Helper function to remove redundant blocks from a valid covering
    def remove_redundant(blocks, uncovered_state=None):
        # We need to check if a block is redundant.
        # A block is redundant if all t-subsets it covers are covered by at least one other block in the set.
        # We can maintain a count of how many times each t-subset is covered by the current set of blocks.
        
        if uncovered_state is not None:
            # If we have the uncovered state from greedy, we know which t-subsets are covered.
            # But we need the counts for redundancy.
            pass
        
        # Efficient way:
        # 1. Count coverage of each t-subset by the current blocks.
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b_idx in blocks:
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            cover_count[all_indices_arr[start:end]] += 1
        
        # 2. Iterate through blocks and check if any t-subset it covers has count 1 (only it covers it).
        # If yes, it's essential. If no, it's redundant.
        new_blocks = []
        for b_idx in blocks:
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            # Check if any of the t-subsets covered by this block have count == 1
            # If all have count > 1, it is redundant.
            is_redundant = True
            for idx in all_indices_arr[start:end]:
                if cover_count[idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it: decrement counts for its t-subsets
                cover_count[all_indices_arr[start:end]] -= 1
            else:
                new_blocks.append(b_idx)
                
        return new_blocks
    
    best_covering = None
    best_size = float('inf')
    
    # Run multiple greedy passes with different seeds
    # How many passes can we do?
    # One pass takes ~150 steps * (50k blocks * 35 ops) ~ 250M ops.
    # In Python, this might take 10-20 seconds per pass.
    # We have ~115 seconds. So maybe 5-10 passes?
    
    # Optimization: We can speed up the greedy step by only considering blocks that cover at least one uncovered t-subset.
    # But identifying them is also costly.
    # Let's just run as many as we can.
    
    seed_counter = 0
    while time.time() < deadline - 10: # Leave 10s for final processing
        blocks, uncovered = run_greedy(seed_counter)
        
        # If we didn't finish due to deadline, repair the rest
        if uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            for idx in uncovered_indices:
                tsub_mask = tsub_masks[idx]
                have = [p for p in range(v) if (tsub_mask >> p) & 1]
                need = k - len(have)
                remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
                if need > 0:
                    extra = random.Random(seed_counter).sample(remaining_points, need)
                else:
                    extra = []
                block = have + extra
                block.sort()
                # Find index of this block in all_k_subsets
                # We need a mask to index map for blocks
                mask = 0
                for p in block:
                    mask |= (1 << p)
                # We didn't precompute mask_to_idx for blocks in this version? 
                # Let's assume we can find it or just add it. 
                # Actually, all_k_subsets contains all possible blocks.
                # We can create a dict for block tuple to index.
                # This is expensive to create every time.
                # Let's create it once outside.
                
                # For now, let's just ensure we have a valid covering.
                # The greedy loop should have covered everything if it ran to completion.
                # If it broke due to deadline, we have a partial covering.
                # We must add blocks for the remaining uncovered t-subsets.
                # We need to map the block to its index in all_k_subsets.
                # Let's create a global lookup if not exists.
                
        # Remove redundant blocks
        optimized_blocks = remove_redundant(blocks)
        
        if len(optimized_blocks) < best_size:
            best_size = len(optimized_blocks)
            best_covering = optimized_blocks
            
        seed_counter += 1
        
    # If we didn't find any covering (should not happen), fallback to simple greedy
    if best_covering is None:
        blocks, _ = run_greedy(0)
        best_covering = blocks
    
    # Final check and conversion
    result = []
    for idx in best_covering:
        result.append(list(all_k_subsets[idx]))
        
    return result

# To make the repair phase in run_greedy work, we need a fast lookup for block indices.
# Let's refactor slightly to include that in the main scope or closure.

# Revised construct to include block lookup
def construct(v, k, t):
    deadline = time.time() + 115.0
    
    # 1. Precomputation
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    
    n_tsub = len(tsub_masks)
    idx_of = {m: i for i, m in enumerate(tsub_masks)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # Map block tuple to index for O(1) lookup during repair
    block_to_idx = {comb: i for i, comb in enumerate(all_k_subsets)}
    
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        subs = []
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.append(idx_of[sm])
        k_sub_tsub_indices.append(subs)
    
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsub, dtype=bool)
        selected_blocks = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            gains = np.zeros(n_ksub, dtype=np.int32)
            for i in range(n_ksub):
                start = offsets_arr[i]
                end = offsets_arr[i+1]
                if start < end:
                    gains[i] = np.sum(uncovered[all_indices_arr[start:end]])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            selected_blocks.append(best_idx)
            
            start = offsets_arr[best_idx]
            end = offsets_arr[best_idx+1]
            uncovered[all_indices_arr[start:end]] = False
            
        return selected_blocks, uncovered

    def remove_redundant(blocks):
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b_idx in blocks:
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            cover_count[all_indices_arr[start:end]] += 1
        
        new_blocks = []
        for b_idx in blocks:
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            is_redundant = True
            for idx in all_indices_arr[start:end]:
                if cover_count[idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                cover_count[all_indices_arr[start:end]] -= 1
            else:
                new_blocks.append(b_idx)
                
        return new_blocks

    best_covering = None
    best_size = float('inf')
    
    seed_counter = 0
    while time.time() < deadline - 5:
        blocks, uncovered = run_greedy(seed_counter)
        
        # Repair if incomplete
        if uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            for idx in uncovered_indices:
                tsub_mask = tsub_masks[idx]
                have = [p for p in range(v) if (tsub_mask >> p) & 1]
                need = k - len(have)
                remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
                if need > 0:
                    extra = random.Random(seed_counter).sample(remaining_points, need)
                else:
                    extra = []
                block = tuple(sorted(have + extra))
                if block in block_to_idx:
                    blocks.append(block_to_idx[block])
                # If block not in block_to_idx, it means we constructed a block not in all_k_subsets?
                # No, all_k_subsets contains all combinations. So it must be there.
                
        optimized_blocks = remove_redundant(blocks)
        
        if len(optimized_blocks) < best_size:
            best_size = len(optimized_blocks)
            best_covering = optimized_blocks
            
        seed_counter += 1
        
    if best_covering is None:
        blocks, _ = run_greedy(0)
        best_covering = blocks
        
    result = []
    for idx in best_covering:
        result.append(list(all_k_subsets[idx]))
        
    return result
