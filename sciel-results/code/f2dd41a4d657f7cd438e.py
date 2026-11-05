import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # All 4-subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Use bitmask representation
    mask_map = {i: 1 << (i-1) for i in range(1, 17)}
    
    def to_mask(s):
        m = 0
        for x in s:
            m |= mask_map[x]
        return m
        
    masks = [to_mask(s) for s in all_sets]
    mask_to_set = {to_mask(s): s for s in all_sets}
    
    # Precompute pairwise properties?
    # For speed, we'll just use bitwise ops.
    
    def check_valid(family, m):
        """
        Check if m can be added to family.
        family is a list of masks.
        """
        n = len(family)
        for i in range(n):
            a = family[i]
            I = a & m
            xor_am = a ^ m
            for j in range(n):
                if i == j:
                    continue
                b = family[j]
                if (b & I) == I and (b & xor_am) == 0:
                    return False
        return True

    def try_greedy(order):
        family = []
        for m in order:
            if check_valid(family, m):
                family.append(m)
        return family

    best_size = 0
    best_family = []
    rng = random.Random(42)
    
    # Phase 1: Random Greedy Restarts
    # We do this for a portion of the time
    greedy_deadline = start_time + 10.0
    
    while time.time() < greedy_deadline:
        shuffled_masks = masks[:]
        rng.shuffle(shuffled_masks)
        family = try_greedy(shuffled_masks)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        if best_size >= 250:
            break

    # Phase 2: Local Search / Random Walk
    # Start from best_family
    family = best_family[:]
    family_set = set(family)
    current_size = len(family)
    
    # Precompute list of indices not in family for quick sampling?
    # Just use masks list and check membership.
    
    # We will perform random moves:
    # 1. Try to add a random valid set (improves size)
    # 2. Try to swap a random set in family with a random set outside (maintains size, explores space)
    
    # To make it faster, we can limit the number of checks per step.
    
    while time.time() < deadline:
        # Try to add
        added = False
        # Pick a random candidate not in family
        # Instead of picking uniformly from all 1820, we can pick from a "candidate pool"
        # But for simplicity, random pick from all.
        
        # Optimization: If we fail to add many times, do a swap.
        
        # Try adding
        for _ in range(10): # Try 10 random adds
            idx = rng.randint(0, n_sets - 1)
            m = masks[idx]
            if m not in family_set:
                if check_valid(family, m):
                    family.append(m)
                    family_set.add(m)
                    current_size += 1
                    added = True
                    break
                    if current_size > best_size:
                        best_size = current_size
                        best_family = family[:]
        
        if not added:
            # Try a swap
            # Pick random A in family, random T not in family
            if len(family) > 0:
                idx_a = rng.randint(0, len(family) - 1)
                a = family[idx_a]
                
                # Try to find a T that is valid in family \ {a}
                # But we want to replace A with T, so T must be valid in family \ {a}
                # And we don't care if T was invalid in family.
                
                # We can just try a few random T's
                swapped = False
                for _ in range(10):
                    idx_t = rng.randint(0, n_sets - 1)
                    t = masks[idx_t]
                    if t == a or t in family_set:
                        continue
                    
                    # Check if t is valid in family without a
                    # We can do this by temporarily removing a, checking, then restoring
                    # Or by checking all pairs in family except a
                    valid = True
                    n = len(family)
                    for i in range(n):
                        if i == idx_a:
                            continue
                        ai = family[i]
                        I = ai & t
                        xor_ait = ai ^ t
                        for j in range(n):
                            if i == j or j == idx_a:
                                continue
                            bj = family[j]
                            if (bj & I) == I and (bj & xor_ait) == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Perform swap
                        family[idx_a] = t
                        family_set.remove(a)
                        family_set.add(t)
                        swapped = True
                        break
                
                # If we swapped, we might now be able to add more?
                # The size is the same. But the structure changed.
                # We continue the loop to try adding.
                
    # Convert best_family back to sorted lists
    result = []
    for m in best_family:
        s = mask_to_set[m]
        result.append(sorted(s))
    
    return result
