import time
import random

def _precompute_neighbors(n=12):
    """
    Precompute bitsets for neighbors in the comparability graph.
    neighbor_bitsets[m] is an integer where bit j is set if m and j are comparable (m != j).
    """
    num_subsets = 1 << n
    neighbor_bitsets = [0] * num_subsets
    
    # For each mask, find all comparable masks
    # m and j are comparable if (m & j) == m or (m & j) == j
    # We can iterate all pairs or use subset/superset enumeration.
    # Given n=12, 4096*4096 = 16.7M checks.
    # This might take ~2-3 seconds in Python.
    
    # Optimization:
    # Subsets of m:
    #   s = m
    #   while s:
    #       neighbor_bitsets[m] |= (1 << s)
    #       s = (s-1) & m
    #   (Include 0? 0 is subset of m. m is superset of 0.)
    #   Wait, strict subset.
    #   So if s == m, skip.
    
    # Supersets of m:
    #   This is harder to enumerate directly without iterating all.
    #   Alternatively, just iterate all j for each m.
    
    start_time = time.time()
    for m in range(num_subsets):
        nb = 0
        for j in range(num_subsets):
            if m == j:
                continue
            if (m & j) == m or (m & j) == j:
                nb |= (1 << j)
        neighbor_bitsets[m] = nb
        
        if time.time() - start_time > 5.0:
            # Fallback if too slow? Unlikely for n=12.
            pass
            
    return neighbor_bitsets

def construct():
    start = time.time()
    total_deadline = 85.0  # Safety margin
    
    n = 12
    m = 40
    num_subsets = 1 << n
    
    # Step 1: Precompute neighbors
    neighbor_bitsets = _precompute_neighbors(n)
    
    # Step 2: Initial Solution
    # Greedy: Start with empty and full, add best next.
    # Or random?
    # Let's use a greedy init based on degrees.
    # Degree of m in the full graph is popcount(neighbor_bitsets[m]).
    
    # We want to pick 40 masks.
    # Let's try a few random restarts with Simulated Annealing.
    
    best_score = -1
    best_masks = [0] * m
    
    # Greedy Init
    current = [0] * m
    used = [False] * num_subsets
    current_bitset = 0
    
    # Pick 0 and num_subsets-1 first?
    # 0 is comparable to all. num_subsets-1 is comparable to all.
    current[0] = 0
    used[0] = True
    current_bitset |= (1 << 0)
    
    current[1] = num_subsets - 1
    used[num_subsets - 1] = True
    current_bitset |= (1 << (num_subsets - 1))
    
    # Greedily fill rest
    for i in range(2, m):
        best_cand = -1
        best_comp = -1
        for cand in range(num_subsets):
            if used[cand]:
                continue
            comp = (current_bitset & neighbor_bitsets[cand]).bit_count()
            if comp > best_comp:
                best_comp = comp
                best_cand = cand
        
        current[i] = best_cand
        used[best_cand] = True
        current_bitset |= (1 << best_cand)
        
    # Calculate initial score
    score = 0
    for i in range(m):
        score += (current_bitset & neighbor_bitsets[current[i]]).bit_count()
    score //= 2 # Each edge counted twice
    
    best_score = score
    best_masks = current[:]
    
    # Step 3: Simulated Annealing
    # Parameters
    T = 5.0
    T_min = 0.1
    alpha = 0.99
    
    t_start = time.time()
    current = best_masks[:]
    current_bitset = 0
    for mask in current:
        current_bitset |= (1 << mask)
    
    # We need to be careful with the bitset size.
    # 1 << 4095 is a huge integer, but Python handles it.
    # Bit operations on 4096-bit integers are fast.
    
    while (time.time() - t_start) < (total_deadline - 2.0):
        # Pick a random index
        idx = random.randint(0, m - 1)
        old_mask = current[idx]
        
        # Pick a random new mask
        # To improve diversity, maybe pick from a neighborhood?
        # Or just random.
        new_mask = random.randint(0, num_subsets - 1)
        
        if new_mask == old_mask:
            continue
        
        # Check if new_mask is already in current (and not at idx)
        # We can check if bit is set in current_bitset.
        # If it is set, and new_mask != old_mask, then it's a duplicate.
        # If new_mask == old_mask, we already skipped.
        # So if (current_bitset >> new_mask) & 1:
        #    if new_mask != old_mask: continue (duplicate)
        #    else: pass (this is the one we are replacing)
        
        if (current_bitset & (1 << new_mask)) and new_mask != old_mask:
            continue
        
        # Calculate delta
        # old_comp: number of OTHER masks comparable to old_mask
        # current_bitset includes old_mask.
        # neighbor_bitsets[old_mask] does NOT include old_mask.
        # So bit_count(current_bitset & neighbor_bitsets[old_mask]) is correct for old_comp.
        
        old_comp = (current_bitset & neighbor_bitsets[old_mask]).bit_count()
        
        # For new_comp, we need to remove old_mask from bitset temporarily?
        # No, new_mask is not in current_bitset (checked above, unless new_mask==old_mask which is skipped).
        # So current_bitset contains old_mask, but not new_mask.
        # We want to count how many masks in (current \ {old_mask}) are comparable to new_mask.
        # Since new_mask != old_mask, and old_mask is in current_bitset,
        # if old_mask is comparable to new_mask, it will be counted.
        # But old_mask is being removed.
        # So we should subtract 1 if old_mask is comparable to new_mask.
        
        # Is old_mask comparable to new_mask?
        # Check: (old_mask & new_mask) == old_mask or (old_mask & new_mask) == new_mask
        # Or simply: is bit old_mask set in neighbor_bitsets[new_mask]?
        
        new_comp = (current_bitset & neighbor_bitsets[new_mask]).bit_count()
        
        # If old_mask is comparable to new_mask, new_comp includes old_mask.
        # But old_mask is leaving. So we should not count it.
        if (neighbor_bitsets[new_mask] & (1 << old_mask)):
            new_comp -= 1
            
        delta = new_comp - old_comp
        
        # Accept?
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            # Accept
            current[idx] = new_mask
            # Update bitset
            current_bitset &= ~(1 << old_mask)
            current_bitset |= (1 << new_mask)
            
            # Update score
            score += delta
            
            if score > best_score:
                best_score = score
                best_masks = current[:]
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Maybe reheat?
            # Or just continue with low T.
            
    # Convert best_masks to lists
    result = []
    for msk in best_masks:
        s = []
        for i in range(n):
            if msk & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
