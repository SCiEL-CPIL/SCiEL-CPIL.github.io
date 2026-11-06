import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s limit
    
    # Generate all 4-element subsets as bitmasks (elements 0-15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    
    def is_sunflower(A, B, C):
        """Check if A, B, C form a sunflower: A∩B = A∩C = B∩C"""
        ab = A & B
        ac = A & C
        bc = B & C
        return ab == ac == bc
    
    def can_add(C, family_masks):
        """Check if C can be added without creating a sunflower with any pair in family."""
        for i in range(len(family_masks)):
            A = family_masks[i]
            for j in range(i + 1, len(family_masks)):
                B = family_masks[j]
                if is_sunflower(A, B, C):
                    return False
        return True
    
    # Greedy initialization with random ordering, multiple trials
    best_family = []
    best_size = 0
    
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        
        random.shuffle(all_masks)
        family = []
        for mask in all_masks:
            if time.time() > deadline - 15:
                break
            if can_add(mask, family):
                family.append(mask)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local search: try to add more sets
    family = best_family[:]
    family_set = set(family)
    
    # Exhaustive add attempt
    added = True
    while added and time.time() < deadline - 10:
        added = False
        for mask in all_masks:
            if mask not in family_set:
                if can_add(mask, family):
                    family.append(mask)
                    family_set.add(mask)
                    added = True
    
    # Swap-based local search
    if time.time() < deadline - 5:
        T = 1.0
        cooling = 0.9995
        
        while time.time() < deadline - 3 and T > 0.01:
            # Try a swap: remove one, add one
            if family:
                idx_out = random.randint(0, len(family) - 1)
                mask_out = family[idx_out]
                
                # Find a random mask not in family
                mask_in = random.choice(all_masks)
                attempts = 0
                while mask_in in family_set and attempts < 50:
                    mask_in = random.choice(all_masks)
                    attempts += 1
                if mask_in in family_set:
                    continue
                
                # Check if mask_in can be added to family without mask_out
                fam_temp = family[:idx_out] + family[idx_out + 1:]
                if can_add(mask_in, fam_temp):
                    new_family = fam_temp + [mask_in]
                    family = new_family
                    family_set.remove(mask_out)
                    family_set.add(mask_in)
                
                # Try pure addition
                if mask_in not in family_set:
                    if can_add(mask_in, family):
                        family.append(mask_in)
                        family_set.add(mask_in)
            
            T *= cooling
        
        # Final attempt to add
        added = True
        while added and time.time() < deadline - 1:
            added = False
            for mask in all_masks:
                if mask not in family_set:
                    if can_add(mask, family):
                        family.append(mask)
                        family_set.add(mask)
                        added = True
    
    # Convert back to sorted lists (1-indexed)
    result = []
    for mask in family:
        elements = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elements)
    
    return result
