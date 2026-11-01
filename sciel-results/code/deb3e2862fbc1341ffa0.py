import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate all words and indices
    # We map word tuple to index 0..q^n-1
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # 2. Precompute coverage masks
    # coverage_mask[i] is an integer where bit j is 1 if word j is within distance 'radius' of word i
    coverage_masks = [0] * total_words
    
    for i, w in enumerate(all_words):
        mask = 0
        # Cover self
        mask |= (1 << i)
        
        if radius == 1:
            # Generate neighbors at distance 1
            for pos in range(n):
                orig_val = w[pos]
                w_list = list(w)
                for new_val in range(q):
                    if new_val != orig_val:
                        w_list[pos] = new_val
                        neighbor = tuple(w_list)
                        j = word_to_idx[neighbor]
                        mask |= (1 << j)
                        w_list[pos] = orig_val # restore
        else:
            # General case: check all (slow, but radius is 1 for this task)
            for j, v in enumerate(all_words):
                if sum(1 for a, b in zip(w, v) if a != b) <= radius:
                    mask |= (1 << j)
        
        coverage_masks[i] = mask
    
    all_ones_mask = (1 << total_words) - 1
    
    # 3. Greedy Set Cover
    uncovered = all_ones_mask
    chosen = []
    chosen_set = set()
    
    # Pre-sort or just iterate? Iterating 729 items is fine.
    # To speed up, we can maintain a list of available candidates?
    # Just iterate all, skip chosen.
    
    while uncovered:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_count = 0
        
        # Optimization: Only consider candidates that cover at least one uncovered bit
        # But we need to count how many.
        for i in range(total_words):
            if i in chosen_set:
                continue
            
            # Count bits in (coverage_masks[i] & uncovered)
            common = coverage_masks[i] & uncovered
            if common:
                count = bin(common).count('1')
                if count > best_count:
                    best_count = count
                    best_idx = i
        
        if best_idx == -1:
            break # Should not happen if uncovered is not 0 and we have valid masks
            
        chosen.append(best_idx)
        chosen_set.add(best_idx)
        uncovered &= ~coverage_masks[best_idx]
        
    # 4. Local Improvement
    # We try to remove or swap codewords to minimize the set size.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Compute M_c = OR of all masks in chosen except c
        # M_c[i] corresponds to chosen[i]
        M_c_list = [0] * len(chosen)
        
        for idx, c in enumerate(chosen):
            mask_c = 0
            for idx2, c2 in enumerate(chosen):
                if idx != idx2:
                    mask_c |= coverage_masks[c2]
            M_c_list[idx] = mask_c
        
        # Try Removal
        removed = False
        for idx, c in enumerate(chosen):
            if M_c_list[idx] == all_ones_mask:
                # Safe to remove
                chosen.pop(idx)
                chosen_set.discard(c)
                improved = True
                removed = True
                break
        
        if removed:
            continue
        
        # Try Swap
        # For each c in chosen, try replacing with c' not in chosen
        swapped = False
        for idx, c in enumerate(chosen):
            base_mask = M_c_list[idx]
            # We need (base_mask | coverage_masks[c_prime]) == all_ones_mask
            # This is equivalent to: all_ones_mask & ~base_mask is a subset of coverage_masks[c_prime]
            # i.e., (all_ones_mask ^ base_mask) & ~coverage_masks[c_prime] == 0
            missing_mask = all_ones_mask & ~base_mask
            
            if missing_mask == 0:
                # This means c is redundant, should have been caught in removal
                # But maybe M_c_list wasn't updated? No, we just computed it.
                # So if missing is 0, removal should have worked.
                pass
            
            for c_prime in range(total_words):
                if c_prime in chosen_set:
                    continue
                
                # Check if c_prime covers all missing bits
                if (missing_mask & ~coverage_masks[c_prime]) == 0:
                    # Swap c with c_prime
                    chosen[idx] = c_prime
                    chosen_set.discard(c)
                    chosen_set.add(c_prime)
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                break
        
        # If no improvement, break
        if not improved:
            break
            
    return [list(all_words[i]) for i in chosen]
