import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    triple_bit = {s: 1 << i for i, s in enumerate(all_subsets)}

    # Precompute blocks and masks
    block_list = []
    block_masks = []
    block_triples = []  # List of lists of subset indices for each block

    for block in itertools.combinations(range(v), k):
        mask = 0
        subs_indices = []
        for sub in itertools.combinations(block, t):
            idx = subset_index[sub]
            mask |= triple_bit[sub]
            subs_indices.append(idx)
        block_list.append(block)
        block_masks.append(mask)
        block_triples.append(subs_indices)

    n_blocks = len(block_list)
    full_mask = (1 << n_subsets) - 1

    def greedy_construct(rng, noise_level=0.0):
        """
        Runs a greedy set cover algorithm.
        If noise_level > 0, it adds random noise to the gain calculation to diversify paths.
        """
        uncovered_mask = full_mask
        selected_indices = []
        used_indices = set()
        
        while uncovered_mask:
            best_gain = -1
            best_candidates = []
            
            for i in range(n_blocks):
                if i in used_indices:
                    continue
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > 0:
                    # Add noise if specified
                    if noise_level > 0:
                        gain += rng.uniform(0, noise_level)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [i]
                    elif gain == best_gain:
                        best_candidates.append(i)
                        
            if not best_candidates:
                break
                
            # Pick randomly from best candidates
            chosen_idx = rng.choice(best_candidates)
            selected_indices.append(chosen_idx)
            used_indices.add(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
        return selected_indices

    def simplify_cover(indices, rng):
        """
        Attempts to remove redundant blocks from the given list of indices.
        Returns a simplified list.
        """
        if not indices:
            return []
            
        # Compute coverage counts
        coverage_counts = [0] * n_subsets
        for idx in indices:
            for sub_idx in block_triples[idx]:
                coverage_counts[sub_idx] += 1
                
        # Iterative removal
        # We maintain a list of active indices
        active = list(indices)
        
        # To allow for order-independence or better removal, we can try multiple passes
        # or shuffle. Let's do a few passes with shuffling to ensure local optimality.
        
        changed = True
        while changed:
            changed = False
            rng.shuffle(active)
            new_active = []
            for idx in active:
                # Check if idx is redundant
                redundant = True
                for sub_idx in block_triples[idx]:
                    if coverage_counts[sub_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for sub_idx in block_triples[idx]:
                        coverage_counts[sub_idx] -= 1
                    changed = True
                else:
                    new_active.append(idx)
            active = new_active
            
        return active

    def local_search(blocks_list, rng, time_budget):
        """
        Takes a list of block tuples (sorted) and tries to improve it via swaps and deletions.
        Returns the improved list.
        """
        if not blocks_list:
            return blocks_list
            
        # Convert to indices for faster lookup? 
        # Since blocks are unique, we can map tuple -> index.
        # But constructing this map for every LS call is expensive.
        # Let's just work with the list of tuples and recompute masks/counts as needed?
        # No, we have global block_masks and block_triples.
        # Let's map the current blocks to their global indices.
        
        # Optimization: Since we generated block_list in a specific order, 
        # and we are passing back blocks from greedy (which are indices), 
        # we should ideally keep indices.
        
        # Let's change the interface: local_search takes a list of global indices.
        # And returns a list of global indices.
        
        # Wait, the greedy_construct returns indices.
        # So we pass indices to local_search.
        
        # Build coverage counts
        coverage_counts = [0] * n_subsets
        for idx in blocks_list:
            for sub_idx in block_triples[idx]:
                coverage_counts[sub_idx] += 1
        
        # Build a set of current blocks for O(1) lookup of "is this block already used"
        # Note: block_list contains tuples.
        # We need to check if a new block (tuple) is already in the set.
        # We can maintain a set of tuples corresponding to the indices in blocks_list.
        current_set = set(block_list[i] for i in blocks_list)
        
        # We also need to be able to find the index of a block if we want to swap?
        # No, we just replace the tuple in the list.
        # But we need the index to update coverage_counts?
        # We can just recompute counts if we do a swap?
        # Recomputing counts for ~70 blocks * 35 triples = 2450 ops. Very fast.
        # So for each swap attempt:
        # 1. Remove old block from counts.
        # 2. Add new block to counts.
        # 3. Check validity.
        # 4. If valid, accept. Else revert.
        
        # To speed up "check validity", we just need to ensure no count becomes 0.
        # We can track the min count? No, just check if any count is 0 after update.
        
        # Let's implement a helper to apply a change and return delta_min_count?
        # Or just check if all counts > 0.
        # Since we only add/remove one block at a time, we can just check the affected triples.
        
        def is_valid_after_change(counts, block_idx, sign):
            # sign = 1 for add, -1 for remove
            for sub_idx in block_triples[block_idx]:
                new_c = counts[sub_idx] + sign
                if new_c < 0:
                    return False
            return True
        
        # Actually, we need to update counts first to check.
        # Let's do:
        # 1. Tentatively update counts.
        # 2. Check if any affected count is 0 (for removal) or if we introduced a 0 (impossible for add).
        # For removal: if any count becomes 0, invalid.
        # For add: always valid (counts increase).
        
        # So for removal: check if any count == 1 before removal.
        # For swap: remove old, check if any count == 1. If ok, add new.
        
        # Let's refine the loop.
        
        n_current = len(blocks_list)
        no_improve = 0
        max_no_improve = 50 # Reset perturbation after this many failed attempts
        
        while time.time() < time_budget:
            # 1. Try to remove a redundant block
            removed = False
            # Pick a random block
            i = rng.randrange(n_current)
            idx = blocks_list[i]
            
            # Check if redundant
            redundant = True
            for sub_idx in block_triples[idx]:
                if coverage_counts[sub_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                for sub_idx in block_triples[idx]:
                    coverage_counts[sub_idx] -= 1
                blocks_list.pop(i)
                current_set.remove(block_list[idx])
                n_current -= 1
                removed = True
                no_improve = 0
                continue # Restart loop to try more removals
            
            if removed:
                continue
                
            # 2. Try to swap a block
            if n_current > 0:
                i = rng.randrange(n_current)
                old_idx = blocks_list[i]
                
                # Identify critical triples (count == 1)
                critical_triples = []
                for sub_idx in block_triples[old_idx]:
                    if coverage_counts[sub_idx] == 1:
                        critical_triples.append(sub_idx)
                        
                if not critical_triples:
                    # Block is redundant? We should have caught it above.
                    # Maybe due to race condition or logic error?
                    # If no critical triples, it's redundant.
                    # Let's just remove it to be safe.
                    for sub_idx in block_triples[old_idx]:
                        coverage_counts[sub_idx] -= 1
                    blocks_list.pop(i)
                    current_set.remove(block_list[old_idx])
                    n_current -= 1
                    no_improve = 0
                    continue
                    
                # The new block MUST cover all critical triples.
                # Let's collect the points involved in critical triples.
                points_needed = set()
                for sub_idx in critical_triples:
                    # sub_idx is an index into all_subsets
                    # We need to get the actual points
                    # all_subsets[sub_idx] is the tuple of points
                    points_needed.update(all_subsets[sub_idx])
                
                if len(points_needed) > k:
                    # Impossible to cover all critical triples with one block
                    no_improve += 1
                    if no_improve > max_no_improve:
                        # Perturbation: Force a random swap or removal?
                        # For now, just reset counter
                        no_improve = 0
                    continue
                
                # We need to find a block that contains `points_needed` and has size k.
                # And is not in `current_set`.
                
                # Heuristic: 
                # Start with points_needed.
                # Fill remaining k - len(points_needed) slots with points that maximize coverage of other triples?
                # Or just try random completions.
                
                # Let's try a few random completions.
                # This is a "repair" strategy.
                
                found_swap = False
                attempts = 10
                for _ in range(attempts):
                    # Start with points_needed
                    new_block_set = set(points_needed)
                    
                    # Fill rest
                    remaining_slots = k - len(new_block_set)
                    if remaining_slots > 0:
                        # Pick random points from v
                        # Exclude points already in new_block_set
                        available = [x for x in range(v) if x not in new_block_set]
                        rng.shuffle(available)
                        for x in available[:remaining_slots]:
                            new_block_set.add(x)
                            
                    new_block_tuple = tuple(sorted(new_block_set))
                    
                    if new_block_tuple in current_set:
                        continue
                    
                    # Check if this block exists in our precomputed list?
                    # Yes, we need the index to get its mask/triples.
                    # We can look it up in a dict?
                    # We didn't build a dict for block_tuple -> index.
                    # Let's build one if we haven't?
                    # Or just calculate the mask on the fly?
                    # Calculating mask on the fly:
                    # mask = 0
                    # for sub in itertools.combinations(new_block_tuple, t):
                    #     mask |= triple_bit[sub]
                    # This is fast (35 ops).
                    
                    # But we also need to check if it's valid.
                    # Validity: After removing old_idx and adding new_block, are all counts > 0?
                    
                    # 1. Remove old_idx
                    for sub_idx in block_triples[old_idx]:
                        coverage_counts[sub_idx] -= 1
                        
                    # 2. Check if any count is 0
                    valid = True
                    for sub_idx in block_triples[old_idx]:
                        if coverage_counts[sub_idx] == 0:
                            valid = False
                            break
                            
                    if not valid:
                        # Revert
                        for sub_idx in block_triples[old_idx]:
                            coverage_counts[sub_idx] += 1
                        continue
                        
                    # 3. Add new_block
                    # Calculate its triples
                    new_triples_indices = []
                    for sub in itertools.combinations(new_block_tuple, t):
                        new_triples_indices.append(subset_index[sub])
                        
                    for sub_idx in new_triples_indices:
                        coverage_counts[sub_idx] += 1
                        
                    # 4. Accept
                    blocks_list[i] = new_block_tuple # Store tuple for now? 
                    # Wait, blocks_list contains indices.
                    # We need to map new_block_tuple back to an index.
                    # If we don't have a map, we can't easily store it as an index.
                    
                    # Solution: Build a map `block_to_idx` at the start.
                    # C(24,7) is 346k. Dict with 346k entries is fine.
                    
                    # Let's assume we have `block_to_idx`.
                    new_idx = block_to_idx.get(new_block_tuple)
                    if new_idx is None:
                        # Should not happen if we generated all blocks
                        # Revert
                        for sub_idx in new_triples_indices:
                            coverage_counts[sub_idx] -= 1
                        for sub_idx in block_triples[old_idx]:
                            coverage_counts[sub_idx] += 1
                        continue
                        
                    # Update state
                    current_set.remove(block_list[old_idx])
                    current_set.add(new_block_tuple)
                    # blocks_list[i] is now new_idx
                    blocks_list[i] = new_idx
                    
                    no_improve = 0
                    found_swap = True
                    break # Try next iteration
                    
                if not found_swap:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        no_improve = 0

        return blocks_list

    # Build the block_to_idx map
    block_to_idx = {b: i for i, b in enumerate(block_list)}

    best_count = float('inf')
    best_blocks = None

    # Run multiple restarts
    # 1. Pure Greedy
    # 2. Noisy Greedy + Local Search
    
    rng = random.Random(42)
    
    # First, a pure greedy to get a baseline
    if time.time() < deadline - 10.0:
        indices = greedy_construct(rng, noise_level=0.0)
        indices = simplify_cover(indices, rng)
        count = len(indices)
        if count < best_count:
            best_count = count
            best_blocks = [block_list[i] for i in indices]

    # Now, try noisy greedy + local search
    # We have limited time.
    # Let's allocate time for a few such runs.
    
    while time.time() < deadline - 10.0:
        # 1. Noisy Greedy
        # Noise level: maybe 0.5? 
        # The max gain is 35.
        # If noise is too high, we pick random blocks.
        # If noise is low, we pick near-best.
        noise = rng.uniform(0.1, 2.0)
        indices = greedy_construct(rng, noise_level=noise)
        
        # 2. Simplify
        indices = simplify_cover(indices, rng)
        
        # 3. Local Search
        # Give it a slice of time
        ls_deadline = time.time() + min(5.0, deadline - time.time() - 5.0)
        indices = local_search(indices, rng, ls_deadline)
        
        # 4. Update best
        count = len(indices)
        if count < best_count:
            best_count = count
            best_blocks = [block_list[i] for i in indices]

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback
        return []
