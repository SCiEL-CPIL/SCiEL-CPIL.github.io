import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees of t-subsets (number of blocks covering each)
    t_subset_degrees = np.array([len(lst) for lst in t_subset_blocks], dtype=np.float64)
    # Inverse degrees for scarcity weighting
    inv_degrees = np.zeros(num_t_subsets, dtype=np.float64)
    mask = t_subset_degrees > 0
    inv_degrees[mask] = 1.0 / t_subset_degrees[mask]
    
    # Precompute initial scarcity weights for each block
    # weight[b] = sum of inv_degrees[t] for t in block
    block_scarcity = np.zeros(num_k_subsets, dtype=np.float64)
    for b_idx, indices in enumerate(block_t_indices):
        if indices:
            block_scarcity[b_idx] = np.sum(inv_degrees[indices])

    def greedy_cover(rng_seed, use_scarcity=True):
        rng = random.Random(rng_seed)
        uncovered = np.ones(num_t_subsets, dtype=np.int8)
        
        # gain[b] = number of uncovered t-subsets covered by block b
        # We maintain this dynamically.
        gain = np.zeros(num_k_subsets, dtype=np.int32)
        for b_idx, indices in enumerate(block_t_indices):
            if indices:
                gain[b_idx] = len(indices)
        
        # active_blocks list for O(1) membership check? Or just a set.
        # A set of active block indices is good for checking if a block is already selected.
        # But we need to iterate over candidates.
        # Strategy:
        # 1. Find the best block (max gain, tie-break by scarcity or random).
        # 2. Add it to solution.
        # 3. Update gains for all blocks that share t-subsets with the selected block.
        
        selected_blocks = []
        active_set = set()
        
        # For speed, we might not want to scan all 50k blocks every time if we can do better.
        # But 50k is small. 150 iterations * 50k = 7.5M ops.
        # Updating gains: For each t-subset in the selected block, iterate its covering blocks.
        # C(7,4)=35 t-subsets per block. Each t-subset is covered by ~1000-2000 blocks.
        # 35 * 2000 = 70k updates per step. 150 * 70k = 10.5M updates.
        # This is very fast in NumPy if vectorized, but loops in Python might be slow.
        # Let's try to optimize the inner loop.
        
        while True:
            # Find best block
            # We want to maximize gain. If tie, maximize scarcity.
            # To avoid scanning all 50k blocks in Python, we can use NumPy argmax?
            # But gain changes.
            # Let's stick to Python scan but optimize it.
            
            max_gain = -1
            best_block = -1
            best_scarcity = -1.0
            candidates = []
            
            # Optimization: If we just pick the first max, it's deterministic.
            # To allow diversity, we can collect candidates with max gain.
            # But collecting a list every time is expensive if the list is large.
            # Alternative: Pick max, then randomly perturb?
            # Let's try: Find max_gain. Then pick a random block with that gain.
            
            # Efficient max finding:
            # Loop is 50k.
            for b_idx in range(num_k_subsets):
                if b_idx in active_set:
                    continue
                g = gain[b_idx]
                if g == 0:
                    continue
                if g > max_gain:
                    max_gain = g
                    best_scarcity = block_scarcity[b_idx]
                    candidates = [b_idx]
                elif g == max_gain:
                    # Tie breaking: prefer higher scarcity
                    s = block_scarcity[b_idx]
                    if s > best_scarcity:
                        best_scarcity = s
                        candidates = [b_idx]
                    elif s == best_scarcity:
                        candidates.append(b_idx)
            
            if max_gain == 0:
                break
                
            best_block = rng.choice(candidates)
            selected_blocks.append(best_block)
            active_set.add(best_block)
            
            # Update gains
            # For each t_idx in block_t_indices[best_block]:
            #   if uncovered[t_idx]:
            #     uncovered[t_idx] = 0
            #     for b_idx in t_subset_blocks[t_idx]:
            #       if b_idx not in active_set:
            #         gain[b_idx] -= 1
            
            # To speed this up, we can precompute the list of t_idx for the block.
            # And iterate.
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = 0
                    for b_idx in t_subset_blocks[t_idx]:
                        if b_idx not in active_set:
                            gain[b_idx] -= 1
                        
        # Minimalization
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        active_blocks_set = set(selected_blocks)
        indices_to_check = list(active_blocks_set)
        rng.shuffle(indices_to_check)
        
        for b_idx in indices_to_check:
            if b_idx not in active_blocks_set:
                continue
            can_remove = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                active_blocks_set.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                    
        return [list(all_k_subsets[b_idx]) for b_idx in active_blocks_set]

    def local_search_polish(blocks, rng_seed, time_limit):
        """
        Try to improve the solution by swapping blocks or removing redundant ones.
        We use a local search approach:
        1. Compute cover counts.
        2. Identify 'critical' blocks (those with unique t-subsets).
        3. Try to replace a critical block with a non-critical block that covers its unique subsets,
           if the replacement allows removing other blocks.
        This is complex. A simpler polish:
        - Randomly remove a block.
        - If coverage breaks, try to add a different block to restore coverage,
          preferring one that allows removing another block.
        """
        rng = random.Random(rng_seed)
        active_blocks = set(range(len(blocks)))
        block_indices = list(range(len(blocks)))
        
        # Map original block index to its t-subsets
        # We need to work with the specific blocks in the solution.
        # Let's re-map to local indices for speed.
        num_local = len(blocks)
        local_block_t_indices = []
        for i, b in enumerate(blocks):
            # b is a list of points. We need t-subsets indices.
            # We can look up from global structures?
            # It's faster to precompute for the local set.
            indices = [t_subset_to_idx[s] for s in itertools.combinations(b, t)]
            local_block_t_indices.append(indices)
            
        # Build local incidence
        local_t_subset_blocks = [[] for _ in range(num_t_subsets)]
        for b_idx, indices in enumerate(local_block_t_indices):
            for idx in indices:
                local_t_subset_blocks[idx].append(b_idx)
                
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx, indices in enumerate(local_block_t_indices):
            for idx in indices:
                cover_count[idx] += 1
                
        start_time = time.time()
        best_size = num_local
        best_blocks_copy = [blocks[i] for i in range(num_local)]
        
        # Simple hill climbing:
        # 1. Try removing a block. If valid, keep it.
        # 2. Try swapping a block for another (not in set).
        
        # We'll do a few passes of removal.
        for pass_num in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Removal pass
            indices_to_check = list(range(num_local))
            rng.shuffle(indices_to_check)
            
            for i in indices_to_check:
                if i not in active_blocks:
                    continue
                can_remove = True
                for t_idx in local_block_t_indices[i]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks.remove(i)
                    for t_idx in local_block_t_indices[i]:
                        cover_count[t_idx] -= 1
                        if cover_count[t_idx] == 0:
                            can_remove = False # Shouldn't happen if check passed
                            break
                            
        # Swap pass:
        # For each active block, try to replace it with a non-active block that covers all its unique t-subsets.
        # This is expensive. Let's try a simpler heuristic:
        # Pick a random active block. Find its unique t-subsets.
        # Find a non-active block that covers all these unique t-subsets.
        # If found, swap. Then minimize.
        
        # Precompute for each active block, its unique t-subsets?
        # This changes as we swap.
        
        # Let's try a limited number of random swaps.
        swaps_attempted = 0
        max_swaps = 500
        
        while swaps_attempted < max_swaps and time.time() - start_time < time_limit:
            if not active_blocks:
                break
            
            # Pick a random active block
            b_idx = rng.choice(list(active_blocks))
            
            # Find unique t-subsets for b_idx
            unique_t = []
            for t_idx in local_block_t_indices[b_idx]:
                if cover_count[t_idx] == 1:
                    unique_t.append(t_idx)
            
            if not unique_t:
                # Block is redundant, remove it
                active_blocks.remove(b_idx)
                for t_idx in local_block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                continue
            
            # Find a candidate block to replace b_idx
            # Candidate must cover all t in unique_t.
            # We can iterate over blocks covering the first unique t, and check if they cover the rest.
            
            first_t = unique_t[0]
            candidates = local_t_subset_blocks[first_t]
            
            found = False
            for cand_idx in candidates:
                if cand_idx in active_blocks:
                    continue
                # Check if cand_idx covers all unique_t
                cand_t_set = set(local_block_t_indices[cand_idx])
                if all(t in cand_t_set for t in unique_t):
                    # Swap
                    active_blocks.remove(b_idx)
                    for t_idx in local_block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    
                    active_blocks.add(cand_idx)
                    for t_idx in local_block_t_indices[cand_idx]:
                        cover_count[t_idx] += 1
                        # Update local_block_t_indices? No, indices are fixed.
                        # But we need to update the list of blocks for t-subsets?
                        # local_t_subset_blocks is static. It maps t -> list of local block indices.
                        # This is correct.
                        
                    found = True
                    break
            
            swaps_attempted += 1
            
            # If we swapped, we might have created redundancies. Minimize again?
            # For speed, we'll just do another removal pass at the end.
            
        # Final minimization
        indices_to_check = list(active_blocks)
        rng.shuffle(indices_to_check)
        for i in indices_to_check:
            if i not in active_blocks:
                continue
            can_remove = True
            for t_idx in local_block_t_indices[i]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                active_blocks.remove(i)
                for t_idx in local_block_t_indices[i]:
                    cover_count[t_idx] -= 1

        if len(active_blocks) < best_size:
            best_size = len(active_blocks)
            best_blocks_copy = [blocks[i] for i in active_blocks]
            
        return best_blocks_copy

    # Run multiple times with different seeds
    best_blocks = None
    best_size = float('inf')
    
    import time
    start_time = time.time()
    max_time = 140 # Leave 10s buffer for the 150s limit
    
    # Phase 1: Greedy construction
    # Run several greedy variants
    num_greedy_runs = 20
    
    for seed in range(num_greedy_runs):
        if time.time() - start_time > max_time:
            break
        
        blocks = greedy_cover(seed)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
    # Phase 2: Local Search Polish on the best result
    if best_blocks is not None:
        time_remaining = max_time - (time.time() - start_time)
        if time_remaining > 5:
            # Try polishing the best solution found
            polished = local_search_polish(best_blocks, 12345, time_remaining - 5)
            if len(polished) < best_size:
                best_size = len(polished)
                best_blocks = polished
    
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_cover(0)
        
    return best_blocks
