import itertools
import time
import random

def construct(v, k, t):
    # Parameters for the specific task C(24, 7, 3)
    # But we should handle general v, k, t if possible, though the task implies specific ones.
    # The evaluator passes v, k, t.
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # If v, k, t are small enough, we can use bitmask approach.
    # C(24, 3) = 2024 triples. Fits in a Python int (arbitrary precision).
    # C(24, 7) = 346104 blocks.
    
    if v > 25 or k > 10:
        # Fallback to simple greedy if parameters are too large for bitmask optimization
        # This is a safeguard, but the task is C(24,7,3).
        return _greedy_construct(v, k, t)

    # 1. Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # 2. Generate all k-blocks and their triple masks
    # For v=24, k=7, this is ~346k blocks.
    # We store: block_indices (list of points), mask (int)
    blocks_data = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        blocks_data.append((block, mask))
        
    # Pre-compute for each triple, which blocks cover it?
    # This helps in the "swap" step.
    # triple_to_blocks[i] = list of block indices j such that block j covers triple i
    triple_to_blocks = [[] for _ in range(num_triples)]
    for j, (block, mask) in enumerate(blocks_data):
        # Iterate set bits in mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(j)
            m ^= lsb
            
    # 3. Greedy Construction
    # Start with empty cover
    current_blocks = [] # List of block indices
    covered_mask = 0
    full_mask = (1 << num_triples) - 1
    
    # Greedy: Pick block with max new coverage
    # To speed up, we can just iterate and pick best each time.
    # O(num_blocks * num_triples) per step is too slow if done naively.
    # But num_blocks is 346k, num_triples 2024.
    # 346k * 2024 ops is ~700M ops per step. Too slow.
    
    # Better Greedy:
    # Maintain list of uncovered triples.
    # For each candidate block, count how many uncovered triples it covers.
    # We can do this efficiently if we iterate over the triples in the block.
    
    # Let's use a standard greedy:
    # 1. Get all uncovered triples.
    # 2. For each block, count intersection with uncovered.
    # 3. Pick max.
    # 4. Update uncovered.
    
    # Optimization:
    # We can pre-compute the list of triples for each block.
    block_triples = []
    for block, mask in blocks_data:
        # Extract indices from mask
        ts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ts.append(idx)
            m ^= lsb
        block_triples.append(ts)
        
    uncovered = set(range(num_triples))
    
    while uncovered:
        best_block_idx = -1
        best_gain = 0
        # Randomize tie-breaking to allow different restarts?
        # For now, deterministic greedy.
        
        # To speed up, we can sample a subset of blocks if time is tight, 
        # but 346k is manageable if we are careful.
        # Iterating 346k blocks and checking intersection with uncovered set.
        # Intersection size calculation:
        # sum(1 for t in block_triples[j] if t in uncovered)
        
        # This is 346k * 35 checks = 12M checks per block selection.
        # If we need 76 blocks, that's 76 * 12M = 912M checks.
        # This might be too slow in Python (10-20 seconds).
        
        # Let's optimize the gain calculation.
        # We can maintain an array `cover_count` for each triple.
        # But for greedy, we just need max gain.
        
        # Alternative: Randomized Greedy.
        # Pick a random uncovered triple.
        # Only consider blocks that cover this triple.
        # Among those, pick the one with max gain.
        # This reduces candidates from 346k to ~346k * (35/2024) ~ 6000 blocks.
        # 6000 * 35 = 210k checks per step.
        # 76 steps * 210k = 16M checks. Very fast.
        
        target_triple = random.choice(list(uncovered))
        candidates = triple_to_blocks[target_triple]
        
        best_gain = -1
        best_block_idx = -1
        
        for j in candidates:
            gain = 0
            for t_idx in block_triples[j]:
                if t_idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = j
                if gain == len(uncovered): # Perfect
                    break
        
        if best_block_idx == -1:
            # Should not happen
            break
            
        current_blocks.append(best_block_idx)
        
        # Update uncovered
        for t_idx in block_triples[best_block_idx]:
            if t_idx in uncovered:
                uncovered.discard(t_idx)
                
        if time.time() > deadline - 20:
            break

    # 4. Local Search: Reduction and Swapping
    # Maintain cover counts
    cover_count = [0] * num_triples
    for j in current_blocks:
        for t_idx in block_triples[j]:
            cover_count[t_idx] += 1
            
    # Helper to check if a set of blocks covers all
    # We will use cover_count array directly.
    
    # Helper to try removing a block
    def try_remove(j):
        # Returns True if removal is valid, and updates cover_count
        # If invalid, reverts cover_count and returns False
        orphaned = []
        for t_idx in block_triples[j]:
            cover_count[t_idx] -= 1
            if cover_count[t_idx] == 0:
                orphaned.append(t_idx)
        
        if not orphaned:
            return True, []
        
        # Revert
        for t_idx in block_triples[j]:
            cover_count[t_idx] += 1
        return False, orphaned

    # Helper to try swapping j_out with j_in
    def try_swap(j_out, j_in):
        # Remove j_out
        orphaned = []
        for t_idx in block_triples[j_out]:
            cover_count[t_idx] -= 1
            if cover_count[t_idx] == 0:
                orphaned.append(t_idx)
        
        if orphaned:
            # Check if j_in covers all orphaned
            # j_in mask must contain all orphaned indices
            # We can check by iterating orphaned and seeing if they are in block_triples[j_in]
            # Or use masks.
            # Since orphaned is small, let's use a set for block_triples[j_in]
            # Pre-compute sets for block_triples? 
            # block_triples is list of lists. 
            # Let's just check:
            in_set = set(block_triples[j_in])
            valid = True
            for t_idx in orphaned:
                if t_idx not in in_set:
                    valid = False
                    break
            
            if valid:
                # Add j_in
                for t_idx in block_triples[j_in]:
                    cover_count[t_idx] += 1
                return True
            else:
                # Revert j_out removal
                for t_idx in block_triples[j_out]:
                    cover_count[t_idx] += 1
                return False
        else:
            # j_out removal was valid, so we can just add j_in? 
            # Wait, if j_out removal was valid, we don't *need* to swap to maintain validity.
            # But swapping might help future removals.
            # However, we want to reduce size. 
            # If removal is valid, we should just remove it, not swap.
            # So this case (orphaned empty) should be handled by try_remove.
            # If we are in try_swap, it implies we *want* to swap.
            # If removal is valid, swap is neutral in size.
            # We only swap if removal is invalid but swap fixes it.
            # So if orphaned is empty, we don't need to swap.
            # Revert
            for t_idx in block_triples[j_out]:
                cover_count[t_idx] += 1
            return False

    # Local Search Loop
    # We try to reduce the number of blocks.
    # 1. Try to remove a block. If valid, do it.
    # 2. If invalid, try to find a swap that makes the cover "better" (e.g., allows future removal).
    #    A good heuristic: Swap with a block that covers the orphaned triples and has high "redundancy" (covers many triples with count > 1).
    
    while time.time() < deadline:
        improved = False
        # Shuffle order of blocks to try removal
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            j = current_blocks[idx]
            
            # Try removal
            valid, orphaned = try_remove(j)
            if valid:
                # Remove it
                current_blocks.pop(idx)
                improved = True
                # Restart loop? Or continue?
                # If we remove, indices are shifted.
                # Let's break and restart the while loop to re-evaluate.
                break
            
            # If invalid, try to find a swap
            # Find blocks that cover all orphaned triples
            # orphaned is a list of triple indices
            # We need intersection of triple_to_blocks[t] for t in orphaned
            
            if len(orphaned) == 1:
                candidates = triple_to_blocks[orphaned[0]]
            elif len(orphaned) == 2:
                # Intersection of two lists
                set1 = set(triple_to_blocks[orphaned[0]])
                set2 = set(triple_to_blocks[orphaned[1]])
                candidates = list(set1.intersection(set2))
            elif len(orphaned) <= 4:
                # Intersection of multiple lists
                # Start with the smallest list
                sets = [set(triple_to_blocks[t]) for t in orphaned]
                sets.sort(key=len)
                candidates = list(sets[0])
                for s in sets[1:]:
                    candidates = [c for c in candidates if c in s]
                    if not candidates:
                        break
            else:
                candidates = []
                
            # Exclude current blocks? No, we can swap with any block.
            # But we should prefer blocks not already in current_blocks to avoid useless swaps?
            # Actually, swapping with a block already in the set is useless.
            # So filter candidates: c not in current_blocks
            
            # Also, we want a "good" swap.
            # Heuristic: Pick candidate that maximizes the number of triples it covers that are currently "over-covered" (count > 1).
            # This increases redundancy, which might allow other blocks to be removed.
            
            best_swap = -1
            best_score = -1
            
            # Limit candidates to speed up
            # If too many, sample
            if len(candidates) > 100:
                candidates = random.sample(candidates, 100)
                
            current_set = set(current_blocks)
            
            for c in candidates:
                if c in current_set:
                    continue
                
                # Calculate score: number of triples in c with cover_count > 1
                # Note: cover_count is currently in the state where j is REMOVED (reverted by try_remove? No, try_remove reverts if invalid)
                # Wait, try_remove reverts the counts if invalid.
                # So cover_count is in the original state (including j).
                # To evaluate swap j -> c:
                # New count for triple t = old_count(t) - (1 if t in j) + (1 if t in c)
                # We want to maximize sum of (new_count(t) > 1) ?
                # Or simply, we want to ensure that after swap, the orphaned triples are covered.
                # We already checked that c covers orphaned.
                # So the swap is valid.
                # To choose the best swap, let's look at how many "extra" coverings it provides.
                
                score = 0
                for t_idx in block_triples[c]:
                    # Current count includes j
                    # New count will be:
                    # if t in j: count - 1 + 1 = count
                    # if t not in j: count + 1
                    # So if t not in j, count increases.
                    # We want to increase counts of triples that are already covered by others?
                    # Actually, we want to create redundancy.
                    # If t is in j, the count doesn't change.
                    # If t is not in j, the count increases by 1.
                    # So we gain redundancy for all t in c that are not in j.
                    # But we only *need* to cover the orphaned ones.
                    # The others are "bonus" redundancy.
                    
                    # Let's just pick the first valid swap or the one with max overlap with j?
                    # Max overlap with j means we replace j with something similar, which might not help.
                    # Min overlap with j means we diversify.
                    
                    # Let's just pick a random valid swap to explore the space.
                    score = 1 
                    break # Just take the first one?
                
                if score > best_score:
                    best_score = score
                    best_swap = c
                    break # Just take the first valid swap to save time
                    
            if best_swap != -1:
                # Perform swap
                # Remove j
                for t_idx in block_triples[j]:
                    cover_count[t_idx] -= 1
                # Add best_swap
                for t_idx in block_triples[best_swap]:
                    cover_count[t_idx] += 1
                # Update current_blocks
                current_blocks[idx] = best_swap
                improved = True
                break # Restart loop
                
        if not improved:
            # No improvement in this pass.
            # Try a random restart? Or just break?
            # If we are stuck, maybe we should restart greedy with a different seed?
            # But we are already in local search.
            # Let's try a few random swaps to escape local minima.
            escaped = False
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                i1 = random.randint(0, len(current_blocks)-1)
                i2 = random.randint(0, len(current_blocks)-1)
                if i1 == i2:
                    continue
                j1 = current_blocks[i1]
                j2 = current_blocks[i2]
                
                # Try swapping j1 with a random block
                # This is complex to do efficiently.
                # Let's just break and accept the current best.
                break

    # Convert indices back to blocks
    result = [blocks_data[j][0] for j in current_blocks]
    return result

def _greedy_construct(v, k, t):
    # Fallback simple greedy
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
