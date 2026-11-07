import time
import random

def _mask_to_list(mask):
    return [i for i in range(12) if (mask >> i) & 1]

def _count_comparables(family_masks):
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (mi & mj) == mi:
                count += 1
            elif (mi & mj) == mj:
                count += 1
    return count

def _calculate_delta(family_masks, idx, new_mask):
    n = len(family_masks)
    old_mask = family_masks[idx]
    old_contrib = 0
    new_contrib = 0
    
    for j in range(n):
        if j == idx:
            continue
        mj = family_masks[j]
        
        if (old_mask & mj) == old_mask:
            old_contrib += 1
        elif (old_mask & mj) == mj:
            old_contrib += 1
            
        if (new_mask & mj) == new_mask:
            new_contrib += 1
        elif (new_mask & mj) == mj:
            new_contrib += 1
            
    return new_contrib - old_contrib

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n_sets = 40
    n_elements = 12
    max_mask = 1 << n_elements
    
    best_score = -1
    best_family = None
    
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        family = set()
        while len(family) < n_sets:
            family.add(random.randint(0, max_mask - 1))
        family_list = list(family)
        
        current_score = _count_comparables(family_list)
        
        if current_score > best_score:
            best_score = current_score
            best_family = family_list[:]
            
        no_improve_count = 0
        max_no_improve = 20000
        
        while no_improve_count < max_no_improve:
            if time.time() > deadline:
                break
            
            idx = random.randint(0, n_sets - 1)
            
            new_mask = random.randint(0, max_mask - 1)
            while new_mask in family:
                new_mask = random.randint(0, max_mask - 1)
            
            delta = _calculate_delta(family_list, idx, new_mask)
            
            if delta > 0:
                old_mask = family_list[idx]
                family.remove(old_mask)
                family.add(new_mask)
                family_list[idx] = new_mask
                current_score += delta
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family_list[:]
            else:
                no_improve_count += 1
                
                if no_improve_count > 1000 and random.random() < 0.05:
                    old_mask = family_list[idx]
                    family.remove(old_mask)
                    family.add(new_mask)
                    family_list[idx] = new_mask
                    current_score = _count_comparables(family_list)
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family_list[:]

    if best_family is None:
        best_family = [i for i in range(n_sets)]
        
    return [_mask_to_list(m) for m in best_family]
