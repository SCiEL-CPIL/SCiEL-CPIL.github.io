import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # Precompute all t-subsets as bitmasks
    all_t_subs = []
    mask_to_idx = {}
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_t_subs.append(mask)
        mask_to_idx[mask] = len(all_t_subs) - 1
    
    num_t_subs = len(all_t_subs)
    
    # Precompute masks for all k-subsets? No, too many (C(19,8) ~ 75582).
    # We will generate candidate blocks on the fly or use a pool.
    
    # Helper to get t-subsets of a block (mask)
    def get_t_subsets_mask(block_mask):
        # Extract points
        points = []
        m = block_mask
        idx = 0
        while m:
            if m & 1:
                points.append(idx)
            m >>= 1
            idx += 1
        # Generate t-subsets
        subs = []
        for comb in itertools.combinations(points, t):
            sub_mask = 0
            for p in comb:
                sub_mask |= (1 << p)
            subs.append(sub_mask)
        return subs

    # 1. Greedy Construction
    def greedy_construct():
        # is_uncovered: True if the t-subset is not yet covered
        # Using a list of bools indexed by mask is not possible directly as mask is up to 2^19.
        # We use a set of uncovered masks or a dict.
        # Better: list of booleans for the num_t_subs indices.
        is_uncovered = [True] * num_t_subs
        
        blocks = []
        # To speed up greedy, we can prioritize points with high degree or just random.
        # Standard greedy: pick block covering most uncovered subsets.
        
        # Initial state: all uncovered
        # We need to efficiently find the best block.
        # For each candidate block, count how many uncovered t-subsets it covers.
        # C(19,8) = 75582 blocks.
        # 75582 * C(8,5)=56 = ~4M operations per step.
        # Number of steps ~ 300. Total ~ 1.2B ops. Might be slow in Python.
        
        # Optimization: Use bitmasks for coverage status?
        # Let's try a simpler greedy: pick a random uncovered subset, extend it to a block.
        # This is faster but might produce a larger covering.
        
        while True:
            # Find first uncovered subset
            first_idx = -1
            for i in range(num_t_subs):
                if is_uncovered[i]:
                    first_idx = i
                    break
            
            if first_idx == -1:
                break
            
            target_mask = all_t_subs[first_idx]
            
            # Extract points from target_mask
            init_points = []
            m = target_mask
            idx = 0
            while m:
                if m & 1:
                    init_points.append(idx)
                m >>= 1
                idx += 1
            
            block_points = list(init_points)
            block_set = set(block_points)
            
            # Grow block to size k
            while len(block_points) < k:
                best_x = -1
                best_gain = -1
                best_ties = []
                
                # Generate (t-1) subsets of current block
                t_minus_1_subs = []
                for comb in itertools.combinations(block_points, t - 1):
                    sub_mask = 0
                    for p in comb:
                        sub_mask |= (1 << p)
                    t_minus_1_subs.append(sub_mask)
                
                # Evaluate gain for each candidate x
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    x_bit = 1 << x
                    for sub_mask in t_minus_1_subs:
                        candidate_mask = sub_mask | x_bit
                        if candidate_mask in mask_to_idx:
                            idx = mask_to_idx[candidate_mask]
                            if is_uncovered[idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_ties = [x]
                    elif gain == best_gain:
                        best_ties.append(x)
                
                if best_x == -1:
                    # No gain, fill randomly
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_x = random.choice(candidates)
                    else:
                        break # Should not happen if v > k
                
                block_points.append(best_x)
                block_set.add(best_x)
            
            block_mask = 0
            for p in block_points:
                block_mask |= (1 << p)
            
            blocks.append(block_mask)
            
            # Update coverage
            for sub_mask in get_t_subsets_mask(block_mask):
                is_uncovered[mask_to_idx[sub_mask]] = False
            
            if time.time() > deadline - 5.0:
                break
        
        return blocks

    blocks = greedy_construct()
    
    # If greedy didn't finish or is invalid, try to fix or return best effort
    # Verify validity
    def is_valid(blocks):
        covered = [False] * num_t_subs
        for b in blocks:
            for sub_mask in get_t_subsets_mask(b):
                covered[mask_to_idx[sub_mask]] = True
        return all(covered)
    
    if not is_valid(blocks):
        # Fallback: simple exhaustive check or return what we have if valid
        # If invalid, we must return a valid one. 
        # The greedy loop breaks on time. If it breaks early, it's invalid.
        # We should ensure we only return valid blocks.
        # If time runs out during greedy, we might have an invalid set.
        # Let's add a check. If invalid, we try to complete it quickly?
        # Or we can just return [] if invalid? No, score 0.
        # Better to ensure greedy finishes. 115s is plenty for 335 blocks.
        # Each step ~ 19 * 56 * 19 ops ~ 20k ops. 335 steps ~ 6M ops. Very fast.
        # The bottleneck is the inner loop of greedy.
        # Let's assume it finishes.
        
        # If it didn't finish, we try to cover remaining subsets with random blocks
        covered = [False] * num_t_subs
        for b in blocks:
            for sub_mask in get_t_subsets_mask(b):
                covered[mask_to_idx[sub_mask]] = True
        
        while True:
            if time.time() > deadline - 2.0:
                break
            found_uncovered = False
            for i in range(num_t_subs):
                if not covered[i]:
                    found_uncovered = True
                    target_mask = all_t_subs[i]
                    init_points = []
                    m = target_mask
                    idx = 0
                    while m:
                        if m & 1:
                            init_points.append(idx)
                        m >>= 1
                        idx += 1
                    
                    block_points = list(init_points)
                    block_set = set(block_points)
                    while len(block_points) < k:
                        candidates = [x for x in range(v) if x not in block_set]
                        if not candidates: break
                        x = random.choice(candidates)
                        block_points.append(x)
                        block_set.add(x)
                    
                    block_mask = 0
                    for p in block_points:
                        block_mask |= (1 << p)
                    blocks.append(block_mask)
                    
                    for sub_mask in get_t_subsets_mask(block_mask):
                        covered[mask_to_idx[sub_mask]] = True
                    break
            if not found_uncovered:
                break
        if not is_valid(blocks):
            return [] # Should not happen if time allows

    # 2. Redundancy Removal
    def remove_redundant(blocks):
        # Calculate coverage counts
        cover_count = [0] * num_t_subs
        for b in blocks:
            for sub_mask in get_t_subsets_mask(b):
                cover_count[mask_to_idx[sub_mask]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    break
                b = blocks[i]
                redundant = True
                for sub_mask in get_t_subsets_mask(b):
                    idx = mask_to_idx[sub_mask]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for sub_mask in get_t_subsets_mask(b):
                        cover_count[mask_to_idx[sub_mask]] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        return blocks, cover_count

    blocks, cover_count = remove_redundant(blocks)
    best_blocks = blocks[:]
    best_len = len(blocks)

    # 3. Simulated Annealing
    # State: blocks (list of masks), cover_count (list of ints)
    # Energy: len(blocks)
    
    # Precompute list of all possible blocks? C(19,8) = 75582.
    # We can generate them on the fly or precompute a list of masks.
    all_blocks = []
    for comb in itertools.combinations(range(v), k):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_blocks.append(mask)
    
    # For fast lookup of which blocks cover a specific t-subset
    # block_covers_subset[block_idx] = list of t-subset indices it covers
    # Or subset_covered_by[t_subset_idx] = list of block indices
    # This is huge. 11628 * ~3000 blocks per subset?
    # C(19,8) / C(19,5) = (19! / (8!11!)) / (19! / (5!14!)) = (5!14!)/(8!11!) = (120 * 14*13*12) / (40320) ...
    # Number of blocks containing a specific 5-set: C(19-5, 8-5) = C(14,3) = 364.
    # So each t-subset is in 364 blocks.
    # Total entries: 11628 * 364 ~ 4.2M. Manageable.
    
    subset_to_blocks = [[] for _ in range(num_t_subs)]
    for b_idx, b_mask in enumerate(all_blocks):
        for sub_mask in get_t_subsets_mask(b_mask):
            t_idx = mask_to_idx[sub_mask]
            subset_to_blocks[t_idx].append(b_idx)
    
    # SA Loop
    current_blocks = blocks[:]
    current_cover_count = cover_count[:]
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 2.0 and T > min_T:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks[:]
        
        # Pick a random block to remove
        if not current_blocks:
            break
            
        i = random.randint(0, len(current_blocks) - 1)
        b_old = current_blocks[i]
        
        # Check if b_old is redundant
        # Find subsets with count 1 that are covered by b_old
        orphans = []
        for sub_mask in get_t_subsets_mask(b_old):
            t_idx = mask_to_idx[sub_mask]
            if current_cover_count[t_idx] == 1:
                orphans.append(t_idx)
        
        if not orphans:
            # Redundant, remove it
            for sub_mask in get_t_subsets_mask(b_old):
                t_idx = mask_to_idx[sub_mask]
                current_cover_count[t_idx] -= 1
            current_blocks.pop(i)
            # Accept improvement
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks[:]
        else:
            # Need to find a replacement block that covers all orphans
            # Find intersection of blocks covering each orphan
            # Start with blocks covering first orphan
            candidate_blocks = set(subset_to_blocks[orphans[0]])
            
            for t_idx in orphans[1:]:
                candidate_blocks.intersection_update(subset_to_blocks[t_idx])
                if not candidate_blocks:
                    break
            
            if candidate_blocks:
                # Pick a random candidate
                # Exclude the block we are removing if it's in the set (it shouldn't be, as it's the only one covering orphans? 
                # Wait, if count is 1, b_old is the ONLY one covering it. So b_old is NOT in subset_to_blocks[t_idx] for orphans?
                # No, b_old IS in subset_to_blocks[t_idx]. But we are removing it.
                # So we must pick a block from candidate_blocks that is NOT b_old.
                # Actually, if b_old is the only one, it's in the list. We need a DIFFERENT block.
                # So we pick from candidate_blocks. If b_old is the only one, candidate_blocks={b_old}.
                # We need a block != b_old.
                
                valid_candidates = [b for b in candidate_blocks if b != i] 
                # Note: i is the index in current_blocks, not all_blocks index.
                # We need to map current_blocks indices to all_blocks indices.
                # This is inefficient. 
                # Let's store (mask, all_block_idx) in current_blocks.
                # Or just use masks.
                # Let's restart with a cleaner data structure for SA.
                pass
            
            # Simpler approach for SA step:
            # 1. Identify orphans.
            # 2. Find all blocks in `all_blocks` that cover all orphans.
            # 3. Exclude the block we are removing.
            # 4. If any exist, pick one randomly.
            # 5. Swap.
            
            if candidate_blocks:
                # Convert set of indices to list
                cand_list = list(candidate_blocks)
                # Filter out the block we are removing
                # We need to know the index of b_old in all_blocks.
                # Let's precompute mask_to_all_block_idx
                mask_to_all_block_idx = {m: i for i, m in enumerate(all_blocks)}
                b_old_global_idx = mask_to_all_block_idx[b_old]
                
                valid_cands = [b for b in cand_list if b != b_old_global_idx]
                
                if valid_cands:
                    new_block_global_idx = random.choice(valid_cands)
                    new_block_mask = all_blocks[new_block_global_idx]
                    
                    # Perform Swap
                    # Remove b_old
                    for sub_mask in get_t_subsets_mask(b_old):
                        t_idx = mask_to_idx[sub_mask]
                        current_cover_count[t_idx] -= 1
                    
                    # Add new_block
                    for sub_mask in get_t_subsets_mask(new_block_mask):
                        t_idx = mask_to_idx[sub_mask]
                        current_cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_block_mask
                    
                    # Check for new redundancies?
                    # We can do a quick local search for deletions after swap
                    # But that's expensive. 
                    # The next SA step will try to remove blocks again.
                    
                    # Accept swap (energy same, but structure changed)
                    # We don't track "energy" for swaps as they don't change block count.
                    # The goal is to enable a deletion.
        
        T *= cooling

    return best_blocks
