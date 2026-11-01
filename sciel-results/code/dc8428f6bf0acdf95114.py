import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different random seeds
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        blocks = _greedy_construct(v, k, t, all_t_subsets, t_sub_map, random_seed=attempts)
        
        # Local Search: Remove redundant blocks
        blocks = _remove_redundant(blocks, v, k, t, t_sub_map)
        
        # Local Search: Swap points to create redundancy
        blocks = _local_search_swap(blocks, v, k, t, t_sub_map, start_time, deadline)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if time.time() > deadline - 5:
            break
            
    return best_blocks if best_blocks else []

def _greedy_construct(v, k, t, all_t_subsets, t_sub_map, random_seed=0):
    random.seed(random_seed)
    uncovered = set(range(len(all_t_subsets)))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered subset to start
        seed_idx = random.choice(list(uncovered))
        seed_subset = all_t_subsets[seed_idx]
        block = list(seed_subset)
        
        while len(block) < k:
            # Find point that maximizes coverage of uncovered subsets
            best_point = -1
            best_gain = -1
            
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
                
            # Precompute combinations of block of size t-1
            block_combos = list(itertools.combinations(block, t-1))
            
            for x in candidates:
                gain = 0
                for R in block_combos:
                    S = tuple(sorted((x,) + R))
                    idx = t_sub_map.get(S)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point == -1:
                best_point = random.choice(candidates)
                
            block.append(best_point)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            idx = t_sub_map[tuple(sub)]
            uncovered.discard(idx)
            
    return blocks

def _remove_redundant(blocks, v, k, t, t_sub_map):
    if not blocks:
        return blocks
        
    # Count occurrences of each t-subset
    counter = {}
    for block in blocks:
        for sub in itertools.combinations(block, t):
            key = t_sub_map[tuple(sub)]
            counter[key] = counter.get(key, 0) + 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            redundant = True
            for sub in itertools.combinations(block, t):
                key = t_sub_map[tuple(sub)]
                if counter.get(key, 0) < 2:
                    redundant = False
                    break
            if redundant:
                for sub in itertools.combinations(block, t):
                    key = t_sub_map[tuple(sub)]
                    counter[key] -= 1
                changed = True
            else:
                new_blocks.append(block)
        blocks = new_blocks
        
    return blocks

def _local_search_swap(blocks, v, k, t, t_sub_map, start_time, deadline):
    if not blocks:
        return blocks
    
    # Maintain counts
    counts = np.zeros(len(t_sub_map), dtype=np.int32)
    # Precompute t-subsets for each block for fast access
    block_t_ids = []
    for block in blocks:
        ids = [t_sub_map[tuple(sub)] for sub in itertools.combinations(block, t)]
        block_t_ids.append(ids)
        for idx in ids:
            counts[idx] += 1
            
    n_blocks = len(blocks)
    
    # We will try to swap points in blocks to make other blocks removable.
    # A block is removable if all its t-subsets have count >= 2.
    # We iterate through blocks and try swaps.
    
    max_iterations = 20000
    iteration = 0
    
    while time.time() < deadline - 5 and iteration < max_iterations:
        iteration += 1
        # Randomly pick a block to modify
        b_idx = random.randint(0, n_blocks - 1)
        block = blocks[b_idx]
        
        # Get current t-subsets and their counts
        # We need to check if swapping p for q is valid.
        # Valid if: All t-subsets in new_block that were unique to old_block are still covered?
        # No, the cover must remain valid.
        # The new block B' covers S'. The old block B covered S.
        # Any S in S \ S' must be covered by at least one OTHER block.
        
        # Let's just try random swaps and check validity.
        # To be efficient, we only try swaps that might increase overlap.
        
        p = random.choice(block)
        candidates = [x for x in range(v) if x not in block]
        if not candidates:
            continue
        q = random.choice(candidates)
        
        new_block = [x for x in block if x != p]
        new_block.append(q)
        new_block.sort()
        
        # Check if new_block is already in blocks? (Should avoid duplicates)
        # For now, ignore duplicate check in local search for speed, 
        # but ensure it's a valid k-subset.
        
        # Calculate loss and gain
        # Lost: t-subsets in old_block containing p
        # Gained: t-subsets in new_block containing q
        
        # We need to ensure that no t-subset in old_block that is NOT in new_block 
        # and has count 1 is left uncovered.
        
        # Efficient check:
        # 1. Identify t-subsets in old_block that contain p.
        # 2. Identify t-subsets in new_block that contain q.
        # 3. The "dangerous" subsets are those in (old_block_subsets - new_block_subsets) 
        #    that have count 1.
        
        # Let's compute the set of t-subsets for old and new.
        # This is O(k^t) which is small (8^5 ~ 32k, but t=5, k=8, C(8,5)=56).
        
        old_subs = set(itertools.combinations(block, t))
        new_subs = set(itertools.combinations(new_block, t))
        
        lost_subs = old_subs - new_subs
        gained_subs = new_subs - old_subs
        
        valid = True
        for sub in lost_subs:
            idx = t_sub_map[tuple(sub)]
            if counts[idx] == 1:
                valid = False
                break
                
        if not valid:
            continue
            
        # If valid, apply the swap
        # Update counts
        for sub in lost_subs:
            idx = t_sub_map[tuple(sub)]
            counts[idx] -= 1
        for sub in gained_subs:
            idx = t_sub_map[tuple(sub)]
            counts[idx] += 1
            
        # Update block
        blocks[b_idx] = new_block
        block_t_ids[b_idx] = [t_sub_map[tuple(sub)] for sub in itertools.combinations(new_block, t)]
        
        # Try to remove redundant blocks
        removed = False
        new_blocks = []
        for i, blk in enumerate(blocks):
            # Check if blk is redundant
            redundant = True
            for idx in block_t_ids[i]:
                if counts[idx] < 2:
                    redundant = False
                    break
            if redundant:
                # Remove
                for idx in block_t_ids[i]:
                    counts[idx] -= 1
                removed = True
            else:
                new_blocks.append(blk)
                # Keep block_t_ids in sync? 
                # It's easier to rebuild block_t_ids if we remove.
        
        if removed:
            # Rebuild block_t_ids to keep indices correct
            blocks = new_blocks
            block_t_ids = []
            for blk in blocks:
                ids = [t_sub_map[tuple(sub)] for sub in itertools.combinations(blk, t)]
                block_t_ids.append(ids)
            n_blocks = len(blocks)
            
    return blocks
