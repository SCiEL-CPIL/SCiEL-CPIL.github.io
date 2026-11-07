import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute all masks
    all_masks = list(range(total_masks))
    
    # Helper to check if a is strict subset of b
    # Using bit manipulation: a is strict subset of b if (a & b) == a and a != b
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    def count_comparable(family_masks):
        count = 0
        k = len(family_masks)
        for i in range(k):
            for j in range(i + 1, k):
                a = family_masks[i]
                b = family_masks[j]
                if is_strict_subset(a, b) or is_strict_subset(b, a):
                    count += 1
        return count

    # 1. Greedy Construction
    # Start with empty set and full set
    family = [0, total_masks - 1]
    in_family = {0, total_masks - 1}
    
    while len(family) < m:
        best_mask = -1
        best_gain = -1
        
        # Find the mask not in family that maximizes comparable pairs with current family
        for mask in all_masks:
            if mask in in_family:
                continue
            
            gain = 0
            for f_mask in family:
                if is_strict_subset(mask, f_mask) or is_strict_subset(f_mask, mask):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
                
        if best_mask == -1:
            break
            
        family.append(best_mask)
        in_family.add(best_mask)
        
        if time.time() > deadline - 10.0:
            break

    # 2. Local Search (Swap)
    # We try to improve the count by swapping one set in family with one set not in family.
    current_score = count_comparable(family)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Shuffle to avoid getting stuck in local optima? 
        # Actually, deterministic swap is fine, but random restarts might help.
        # Let's do a few random restarts if time permits.
        
        # Try all swaps
        # To speed up, we can just iterate.
        for i in range(len(family)):
            if time.time() > deadline - 5.0:
                break
            old_mask = family[i]
            # Calculate loss of removing old_mask
            # Loss is number of sets in family (excluding old_mask) comparable to old_mask
            loss = 0
            for j in range(len(family)):
                if i == j:
                    continue
                if is_strict_subset(old_mask, family[j]) or is_strict_subset(family[j], old_mask):
                    loss += 1
            
            # Try replacing with a mask not in family
            # We can sample or iterate all. Iterating all 4096 is fast enough.
            for new_mask in all_masks:
                if new_mask in in_family:
                    continue
                
                # Calculate gain of adding new_mask
                # Gain is number of sets in family (excluding old_mask) comparable to new_mask
                gain = 0
                for j in range(len(family)):
                    if i == j:
                        continue
                    if is_strict_subset(new_mask, family[j]) or is_strict_subset(family[j], new_mask):
                        gain += 1
                
                delta = gain - loss
                if delta > 0:
                    # Perform swap
                    family[i] = new_mask
                    in_family.remove(old_mask)
                    in_family.add(new_mask)
                    current_score += delta
                    improved = True
                    # Restart the inner loop or just continue?
                    # Usually, after a swap, other swaps might become valid/invalid.
                    # Breaking to re-evaluate all is safer but slower.
                    # Let's break out of the new_mask loop to restart the i loop?
                    # Or just continue. Since we updated family[i], the 'loss' for other i's changes.
                    # It's better to restart the whole while loop or at least the i loop.
                    break
            
            if improved:
                # Restart the i loop to recalculate losses with the new family composition
                # Actually, just breaking the new_mask loop isn't enough, we should restart the i loop
                # or the while loop. Let's restart the while loop for correctness and simplicity.
                # It might be slow if many swaps happen, but N=40 is small.
                break

    # 3. Final Verification and Formatting
    # Ensure distinctness and types
    final_sets = []
    seen = set()
    for mask in family:
        if mask not in seen:
            seen.add(mask)
            s = [b for b in range(n) if (mask >> b) & 1]
            final_sets.append(s)
            
    # If we didn't get enough (shouldn't happen), pad with random sets? 
    # The problem says "exactly forty distinct subsets".
    # Greedy always finds 40.
    if len(final_sets) < m:
        # Fallback: fill with random sets not in seen
        for mask in all_masks:
            if mask not in seen:
                seen.add(mask)
                s = [b for b in range(n) if (mask >> b) & 1]
                final_sets.append(s)
                if len(final_sets) == m:
                    break
    
    # Convert to lists of ints
    # The problem requires lists of distinct integers.
    # My construction does this.
    
    return final_sets[:m]

# EVOLVE-BLOCK-START
P = {'n': 12, 'm': 40}

# The construct function defined above is the one to be used.
# However, the prompt asks to keep the function names and signatures.
# The seed program has `def construct():`.
# I will place the logic inside `construct`.

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute all masks
    all_masks = list(range(total_masks))
    
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    def count_comparable(family_masks):
        count = 0
        k = len(family_masks)
        for i in range(k):
            for j in range(i + 1, k):
                a = family_masks[i]
                b = family_masks[j]
                if is_strict_subset(a, b) or is_strict_subset(b, a):
                    count += 1
        return count

    # 1. Greedy Construction
    family = [0, total_masks - 1]
    in_family = {0, total_masks - 1}
    
    while len(family) < m:
        best_mask = -1
        best_gain = -1
        
        for mask in all_masks:
            if mask in in_family:
                continue
            
            gain = 0
            for f_mask in family:
                if is_strict_subset(mask, f_mask) or is_strict_subset(f_mask, mask):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
                
        if best_mask == -1:
            break
            
        family.append(best_mask)
        in_family.add(best_mask)
        
        if time.time() > deadline - 10.0:
            break

    # 2. Local Search (Swap)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(len(family)):
            if time.time() > deadline - 5.0:
                break
            old_mask = family[i]
            
            loss = 0
            for j in range(len(family)):
                if i == j:
                    continue
                if is_strict_subset(old_mask, family[j]) or is_strict_subset(family[j], old_mask):
                    loss += 1
            
            for new_mask in all_masks:
                if new_mask in in_family:
                    continue
                
                gain = 0
                for j in range(len(family)):
                    if i == j:
                        continue
                    if is_strict_subset(new_mask, family[j]) or is_strict_subset(family[j], new_mask):
                        gain += 1
                
                delta = gain - loss
                if delta > 0:
                    family[i] = new_mask
                    in_family.remove(old_mask)
                    in_family.add(new_mask)
                    improved = True
                    break
            
            if improved:
                break

    # 3. Final Formatting
    final_sets = []
    seen = set()
    for mask in family:
        if mask not in seen:
            seen.add(mask)
            s = [b for b in range(n) if (mask >> b) & 1]
            final_sets.append(s)
            
    if len(final_sets) < m:
        for mask in all_masks:
            if mask not in seen:
                seen.add(mask)
                s = [b for b in range(n) if (mask >> b) & 1]
                final_sets.append(s)
                if len(final_sets) == m:
                    break
    
    return final_sets[:m]
# EVOLVE-BLOCK-END
