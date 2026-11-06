import random
import math
import time

def construct():
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples as masks and lists
    all_triples = []
    triple_masks = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                all_triples.append((i, j, k))
                triple_masks[mask] = [i, j, k]

    def is_valid(triple, existing_masks):
        """Check if triple (as mask) is linear with existing triples (list of masks)."""
        t_mask = triple
        for e_mask in existing_masks:
            if (t_mask & e_mask).bit_count() > 1:
                return False
        return True

    def get_independence_number_and_set(triple_masks_list):
        """
        Returns (max_size, mask_of_max_set).
        triple_masks_list is a list of integer masks.
        """
        max_size = 0
        best_mask = 0
        
        # Iterate all subsets
        # Optimization: Skip masks that are subsets of already checked independent masks? 
        # Hard to do efficiently without a lattice.
        # Just iterate all.
        
        # We can prune: if popcount(mask) <= max_size, skip.
        # But we need to find the max.
        
        # A faster way:
        # If we have a set of forbidden masks (triples), a set S is independent 
        # if it contains no forbidden mask.
        
        # Let's just loop. 2^16 = 65536.
        # Inner loop is len(triple_masks_list) ~ 30-40.
        # 65536 * 40 = 2.6M operations.
        # In Python, this is ~0.2s.
        # We need many iterations.
        
        # Can we use NumPy?
        # Create an array of all masks?
        # masks = np.arange(1 << n)
        # popcounts = np.array([bin(m).count('1') for m in masks]) # Slow
        # 
        # Let's stick to Python but optimize the inner loop.
        
        t_masks = triple_masks_list
        if not t_masks:
            return n, (1 << n) - 1

        # Sort t_masks? No.
        
        max_size = 0
        best_mask = 0
        
        # We can iterate masks in descending order of popcount?
        # No, we just need the max.
        
        # Trick: 
        # If mask is independent, all subsets are independent.
        # If mask is NOT independent, supersets might be independent? No, if it contains a triple, supersets contain it too.
        # So, if mask is dependent, all supersets are dependent.
        # This suggests a DFS/Branch and Bound might be faster?
        # But for n=16, brute force is predictable.
        
        # Let's try to speed up the check using bit operations.
        # For a given mask m, check if any t in t_masks is subset of m.
        # (m & t) == t
        
        for m in range(1 << n):
            # Quick check: if popcount(m) <= max_size, skip
            if m.bit_count() <= max_size:
                continue
                
            is_ind = True
            for t in t_masks:
                if (m & t) == t:
                    is_ind = False
                    break
            
            if is_ind:
                max_size = m.bit_count()
                best_mask = m
                
        return max_size, best_mask

    # Initial state: Greedy construction
    initial_triples = []
    initial_masks = []
    
    # Randomize order
    shuffled_triples = all_triples[:]
    random.shuffle(shuffled_triples)
    
    for t in shuffled_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        if is_valid(mask, initial_masks):
            initial_triples.append(t)
            initial_masks.append(mask)
            if len(initial_triples) >= 37:
                break

    best_triples = initial_triples[:]
    best_masks = initial_masks[:]
    best_alpha, _ = get_independence_number_and_set(best_masks)
    
    current_triples = initial_triples[:]
    current_masks = initial_masks[:]
    current_alpha, current_max_set_mask = get_independence_number_and_set(current_masks)
    
    start_time = time.time()
    timeout = 115
    
    T = 1.0
    cooling = 0.999
    
    # Precompute list of masks for all triples for fast lookup?
    # We have triple_masks dict.
    
    while time.time() - start_time < timeout:
        # 1. Identify candidates in the current max independent set
        # current_max_set_mask contains the points of the largest independent set found in last eval
        # But we need to ensure it's still valid for current_masks?
        # It is, because we just computed it.
        
        # Find all triples contained in current_max_set_mask
        candidates_in_set = []
        for t in all_triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            if (mask & current_max_set_mask) == mask:
                # Check if it's already in current_triples
                # We can check if mask is in current_masks
                # O(M) check.
                if mask not in current_masks:
                    # Check validity
                    if is_valid(mask, current_masks):
                        candidates_in_set.append((t, mask))
        
        move_type = None
        new_triples = current_triples[:]
        new_masks = current_masks[:]
        
        if candidates_in_set:
            # High priority: Add a triple from the independent set
            # This should reduce alpha
            t, mask = random.choice(candidates_in_set)
            new_triples.append(t)
            new_masks.append(mask)
            move_type = 'add_attack'
        else:
            # No direct attack possible. Need to escape.
            # Random move: Remove, Add, Swap
            r = random.random()
            if r < 0.4:
                # Remove random
                if new_masks:
                    idx = random.randint(0, len(new_masks) - 1)
                    new_triples.pop(idx)
                    new_masks.pop(idx)
                    move_type = 'remove'
            elif r < 0.7:
                # Add random valid triple
                added = False
                for _ in range(20):
                    t = random.choice(all_triples)
                    mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                    if mask not in new_masks:
                        if is_valid(mask, new_masks):
                            new_triples.append(t)
                            new_masks.append(mask)
                            added = True
                            break
                if added:
                    move_type = 'add_random'
                else:
                    continue
            else:
                # Swap
                if not new_masks:
                    continue
                idx = random.randint(0, len(new_masks) - 1)
                old_t = new_triples.pop(idx)
                old_mask = new_masks.pop(idx)
                added = False
                for _ in range(20):
                    t = random.choice(all_triples)
                    mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                    if mask != old_mask and mask not in new_masks:
                        if is_valid(mask, new_masks):
                            new_triples.append(t)
                            new_masks.append(mask)
                            added = True
                            break
                if added:
                    move_type = 'swap'
                else:
                    # Revert
                    new_triples.insert(idx, old_t)
                    new_masks.insert(idx, old_mask)
                    continue
        
        # Evaluate new state
        new_alpha, new_max_set_mask = get_independence_number_and_set(new_masks)
        
        # Acceptance
        if new_alpha < current_alpha:
            current_triples = new_triples
            current_masks = new_masks
            current_alpha = new_alpha
            current_max_set_mask = new_max_set_mask
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = new_triples[:]
                best_masks = new_masks[:]
        else:
            delta = new_alpha - current_alpha
            if delta == 0:
                if random.random() < 0.5:
                    current_triples = new_triples
                    current_masks = new_masks
                    current_max_set_mask = new_max_set_mask
            else:
                # delta > 0
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_triples = new_triples
                    current_masks = new_masks
                    current_alpha = new_alpha
                    current_max_set_mask = new_max_set_mask
        
        T *= cooling

    return best_triples
