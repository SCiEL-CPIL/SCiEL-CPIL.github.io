# EVOLVE-BLOCK-START
import time

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    m = len(sets)
    cnt = 0
    for i in range(m):
        si = sets[i]
        for j in range(i + 1, m):
            sj = sets[j]
            if si and sj:
                if len(si) < len(sj):
                    is_sub = True
                    s_j_set = set(sj)
                    for x in si:
                        if x not in s_j_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
                elif len(sj) < len(si):
                    is_sub = True
                    s_i_set = set(si)
                    for x in sj:
                        if x not in s_i_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
            elif si and not sj:
                cnt += 1
            elif not si and sj:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    # Precompute comparability bitsets
    # comp_bitset[i] is an integer where bit j is 1 if mask i and mask j are comparable
    comp_bitset = [0] * total_masks
    
    # Optimization: Precompute popcount for small integers? No, Python int.bit_count() is fast.
    
    # We can compute comp_bitset efficiently.
    # Two masks a, b are comparable if (a & b) == a or (a & b) == b.
    # Since total_masks is 4096, double loop is 16M iterations.
    # In Python, 16M iterations might take ~1-2 seconds. Acceptable.
    
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                ci |= (1 << j)
        comp_bitset[i] = ci
        
    # Greedy Construction
    selected = []
    selected_mask = 0 # Bitmask of selected mask IDs
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        # Find mask that maximizes comparability with current selected
        for k in range(total_masks):
            if (selected_mask >> k) & 1:
                continue
            # Count comparable pairs with current selected
            # We need count of selected masks comparable to k
            # This is popcount(comp_bitset[k] & selected_mask)
            score = (comp_bitset[k] & selected_mask).bit_count()
            if score > best_score:
                best_score = score
                best_mask = k
        
        selected.append(best_mask)
        selected_mask |= (1 << best_mask)
        
    # Calculate initial score
    current_score = 0
    for i in range(m):
        current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
    current_score //= 2
    
    best_score = current_score
    best_selected = list(selected)
    
    # Local Search
    # We will try to swap one selected mask with one non-selected mask
    # Delta = gain(new) - loss(old)
    # gain(new) = popcount(comp_bitset[new] & (selected_mask without old))
    # loss(old) = popcount(comp_bitset[old] & (selected_mask without old))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through all selected masks to remove
        for idx in range(m):
            if time.time() >= deadline:
                break
            
            old_mask = selected[idx]
            # Mask of selected sets excluding old_mask
            sel_without_old = selected_mask & ~(1 << old_mask)
            
            # Calculate loss
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
            
            # Try adding all non-selected masks
            # We can iterate 0..4095 and skip selected ones
            # Or iterate only non-selected ones.
            # For speed, iterating all 4096 and checking bit is fast enough.
            for new_mask in range(total_masks):
                if (selected_mask >> new_mask) & 1:
                    continue
                
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                
                if delta > 0:
                    # Accept swap
                    selected[idx] = new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    
                    improved = True
                    # Restart scanning from the beginning to ensure we don't miss better swaps
                    # Breaking out of new_mask loop and restarting idx loop
                    # Actually, breaking both is safer for "hill climbing"
                    break
        
    # Convert best_selected to lists of integers
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure types are correct
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
        
    return final_result
# EVOLVE-BLOCK-END
