def construct():
    import random
    import time
    from itertools import combinations
    
    # Represent 4-subsets of [1,16] as 16-bit integers
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n = len(all_sets)  # 1820
    
    # Precompute pairwise intersections as bitmasks
    # inter[i][j] = all_sets[i] & all_sets[j]
    # This is 1820*1820 = 3.3M entries, each an int - feasible
    
    inter = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            val = all_sets[i] & all_sets[j]
            inter[i][j] = val
            inter[j][i] = val
    
    best_family = []
    start_time = time.time()
    timeout = 55
    
    best_size = 0
    best_result = []
    
    for trial in range(100):
        if time.time() - start_time > timeout:
            break
        
        random.seed(trial * 12345 + 67890)
        indices = list(range(n))
        random.shuffle(indices)
        
        family = []
        family_indices = []
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            
            C = all_sets[idx]
            can_add = True
            
            # For each pair in family, check sunflower condition
            for i in range(len(family_indices)):
                ai = family_indices[i]
                inter_AC = inter[ai][idx]
                for j in range(i+1, len(family_indices)):
                    bj = family_indices[j]
                    if inter[ai][bj] == inter_AC and inter[ai][bj] == inter[bj][idx]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                family.append(C)
                family_indices.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_result = family
            if best_size >= 250:
                break
    
    # Convert back to sorted lists
    result = []
    for mask in best_result:
        elems = []
        for x in range(16):
            if mask & (1 << x):
                elems.append(x + 1)
        result.append(sorted(elems))
    
    return result
