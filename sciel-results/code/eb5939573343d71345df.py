import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 50.0
    
    n = 16
    k = 4
    all_sets = [set(c) for c in combinations(range(1, n + 1), k)]
    num_sets = len(all_sets)
    
    masks = [0] * num_sets
    for i, s in enumerate(all_sets):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        masks[i] = m
        
    best_family = []
    
    def can_add(candidate_idx, family_indices, index_by_core):
        cand_mask = masks[candidate_idx]
        for a_idx in family_indices:
            a_mask = masks[a_idx]
            core_mask = cand_mask & a_mask
            p_union = a_mask ^ cand_mask
            
            if core_mask in index_by_core:
                for b_idx in index_by_core[core_mask]:
                    b_mask = masks[b_idx]
                    if (b_mask & p_union) == 0 and (b_mask & core_mask) == core_mask:
                        return False
        return True

    def add_to_index(idx, index_by_core):
        b_set = all_sets[idx]
        for r in range(0, 4):
            for sub in combinations(b_set, r):
                sm = 0
                for x in sub:
                    sm |= (1 << (x - 1))
                if sm not in index_by_core:
                    index_by_core[sm] = []
                index_by_core[sm].append(idx)

    indices = list(range(num_sets))
    
    # Adaptive restarts
    restarts = 0
    while time.time() - start_time < time_limit:
        random.shuffle(indices)
        family = []
        index_by_core = {}
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            if can_add(idx, family, index_by_core):
                family.append(idx)
                add_to_index(idx, index_by_core)
                
        if len(family) > len(best_family):
            best_family = family[:]
            # If we found a very good one, we might stop early?
            # No, keep searching.
            
        restarts += 1

    result = [sorted(all_sets[i]) for i in best_family]
    
    if len(result) > 250:
        result = result[:250]
        
    return result
