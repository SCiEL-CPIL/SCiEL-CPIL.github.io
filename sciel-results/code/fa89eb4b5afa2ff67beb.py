# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 16
    k = 4
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
    
    # 1. Deterministic Greedy Baseline
    family_masks = []
    for s_mask in all_masks:
        conflict = False
        for i in range(len(family_masks)):
            a = family_masks[i]
            for j in range(i + 1, len(family_masks)):
                b = family_masks[j]
                if (a & b) == (a & s_mask) == (b & s_mask):
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            family_masks.append(s_mask)
            
    best_family = family_masks[:]
    best_size = len(family_masks)
    
    # 2. Local Search / Simulated Annealing
    # We will try to improve the family by swapping elements or adding new ones.
    # State: family_masks (list of masks in the family)
    # Complement: set of masks not in family
    
    start_time = time.time()
    time_limit = 50.0
    
    # Use a set for O(1) lookup of what's in the family?
    # But we need to iterate over family for conflict check.
    # Let's maintain a list `current_family` and a set `family_set`.
    
    current_family = best_family[:]
    family_set = set(current_family)
    
    # Pre-shuffle complement for random access?
    # Or just pick random indices from all_masks and check if in family_set.
    
    while time.time() - start_time < time_limit:
        # Try to add a random set
        # Pick 100 random candidates
        added_any = False
        for _ in range(100):
            idx = random.randint(0, len(all_masks) - 1)
            s_mask = all_masks[idx]
            if s_mask in family_set:
                continue
                
            conflict = False
            for i in range(len(current_family)):
                a = current_family[i]
                for j in range(i + 1, len(current_family)):
                    b = current_family[j]
                    if (a & b) == (a & s_mask) == (b & s_mask):
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                current_family.append(s_mask)
                family_set.add(s_mask)
                added_any = True
                if len(current_family) > best_size:
                    best_size = len(current_family)
                    best_family = current_family[:]
        
        # If we added something, continue. Otherwise, try swaps.
        if not added_any:
            # Try a swap: Remove one, add one.
            # Pick a random element to remove
            if current_family:
                rem_idx = random.randint(0, len(current_family) - 1)
                rem_mask = current_family[rem_idx]
                
                # Remove it
                current_family.pop(rem_idx)
                family_set.discard(rem_mask)
                
                # Try to add a random replacement
                found_replacement = False
                for _ in range(50):
                    idx = random.randint(0, len(all_masks) - 1)
                    s_mask = all_masks[idx]
                    if s_mask in family_set:
                        continue
                    
                    conflict = False
                    for i in range(len(current_family)):
                        a = current_family[i]
                        for j in range(i + 1, len(current_family)):
                            b = current_family[j]
                            if (a & b) == (a & s_mask) == (b & s_mask):
                                conflict = True
                                break
                        if conflict:
                            break
                    
                    if not conflict:
                        current_family.append(s_mask)
                        family_set.add(s_mask)
                        found_replacement = True
                        if len(current_family) > best_size:
                            best_size = len(current_family)
                            best_family = current_family[:]
                        break
                
                # If we didn't find a replacement, put the old one back
                if not found_replacement:
                    current_family.append(rem_mask)
                    family_set.add(rem_mask)
                    
    # Convert best_family to lists of integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
