import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin

    # Bitset of reachable sums. Bit i is 1 if sum i is reachable.
    # We care about sums up to 200.
    MASK = (1 << 201) - 1  # Bits 0 to 200 inclusive
    
    def is_valid_addition(reachable, x):
        """Check if adding x makes 200 reachable."""
        new_reachable = reachable | (reachable << x)
        new_reachable &= MASK
        return not (new_reachable & (1 << 200))
    
    def get_reachable(selected):
        """Compute reachable bitset from a list of selected numbers."""
        reachable = 1
        for x in selected:
            reachable |= (reachable << x)
            reachable &= MASK
        return reachable

    # 1. Greedy Initialization: Descending order
    selected = []
    reachable = 1
    
    for x in range(100, 0, -1):
        if is_valid_addition(reachable, x):
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= MASK
        # else: skip x

    # 2. Local Search: Try to add back excluded elements
    # We need to know which elements are excluded.
    selected_set = set(selected)
    excluded = [x for x in range(1, 101) if x not in selected_set]
    
    # Try single additions
    changed = True
    while changed and (time.time() - start_time) < deadline:
        changed = False
        # Re-check all excluded, as adding one might allow another? 
        # No, adding an element only increases reachable sums, so if it was invalid before, 
        # it remains invalid unless we remove something else. 
        # So simple "add if valid" works for single additions.
        for x in excluded[:]: # iterate over a copy
            if is_valid_addition(reachable, x):
                selected.append(x)
                selected_set.add(x)
                excluded.remove(x)
                reachable |= (reachable << x)
                reachable &= MASK
                changed = True
    
    # 3. Local Search: Swaps (Remove 1, Add 1) or (Remove 1, Add 2)
    # To maximize cardinality, we look for opportunities to swap a selected element 
    # for one or more excluded elements, increasing the total count.
    
    # Try Remove 1, Add 2 (net gain +1)
    # This is more complex. Let's try a simple random restart or hill climbing.
    # Given the small size (100 elements), we can try all pairs of (remove, add1, add2).
    # But that's O(N^3). 100^3 = 1,000,000. Each check is O(1) bit op. 
    # This is feasible within 80 seconds.
    
    # Let's structure the search for improvements:
    # Current size: len(selected)
    
    # Helper to check validity of a set of changes
    def try_change(selected_list, excluded_list, remove_idx, add_indices):
        """
        Simulates removing selected_list[remove_idx] and adding elements at add_indices from excluded_list.
        Returns (new_selected, new_reachable, is_valid)
        """
        # This is expensive to copy lists every time. 
        # Better to update bitset incrementally?
        # Removing from bitset is hard (subset sum is not invertible easily).
        # So we must recompute or use a different representation.
        # Given N=100, recomputing reachable from scratch takes 100 steps.
        # 100 steps * 1,000,000 checks = 100,000,000 operations. 
        # Python might be slow for 100M ops in 80s. 
        # Let's optimize: Only try promising swaps.
        
        pass

    # Alternative: Since the greedy solution is likely near-optimal, 
    # simple single-additions might be sufficient. 
    # Let's add a randomized local search for swaps.
    
    import random
    random.seed(42)
    
    best_selected = selected[:]
    best_size = len(selected)
    best_reachable = reachable
    
    # Randomized local search
    for _ in range(5000): # Limit iterations to ensure speed
        if (time.time() - start_time) > deadline:
            break
            
        # Pick a random selected element to remove
        if not best_selected:
            break
        rem_idx = random.randint(0, len(best_selected) - 1)
        rem_val = best_selected[rem_idx]
        
        # Calculate reachable without rem_val
        # We have to recompute. 
        # To speed this up, we can precompute prefix/suffix bitsets? 
        # Or just recompute. 100 elements * 5000 iters = 500,000 recomputes.
        # Each recompute is 100 bit shifts. 50M bit shifts. Fast enough.
        
        temp_selected = best_selected[:]
        temp_selected.pop(rem_idx)
        
        temp_reachable = 1
        for x in temp_selected:
            temp_reachable |= (temp_reachable << x)
            temp_reachable &= MASK
        
        # Try to add random excluded elements
        # We want to add more elements than we removed.
        # Let's try adding 2 random excluded elements.
        
        # Get current excluded list
        best_set = set(best_selected)
        current_excluded = [x for x in range(1, 101) if x not in best_set]
        
        if len(current_excluded) < 2:
            continue
            
        # Pick 2 random candidates
        cand1_idx = random.randint(0, len(current_excluded) - 1)
        cand2_idx = random.randint(0, len(current_excluded) - 1)
        if cand1_idx == cand2_idx:
            continue
        cand1 = current_excluded[cand1_idx]
        cand2 = current_excluded[cand2_idx]
        
        # Check if adding cand1 and cand2 to temp_reachable is valid
        r1 = temp_reachable | (temp_reachable << cand1)
        r1 &= MASK
        if r1 & (1 << 200):
            continue
            
        r2 = r1 | (r1 << cand2)
        r2 &= MASK
        if r2 & (1 << 200):
            continue
            
        # Valid improvement: Removed 1, Added 2. Net +1 size.
        new_selected = temp_selected + [cand1, cand2]
        best_selected = new_selected
        best_reachable = r2
        best_size += 1

    # Final check and cleanup
    # Ensure nonempty
    if not best_selected:
        # Fallback: A single element {1} is valid.
        return [1]
        
    return best_selected
