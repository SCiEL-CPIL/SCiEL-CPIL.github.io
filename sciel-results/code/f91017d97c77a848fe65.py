def construct():
    import time
    from itertools import combinations
    import random

    # 1. Setup
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n = len(all_sets) # 1820
    mask_to_idx = {mask: i for i, mask in enumerate(all_sets)}
    
    # Precompute conflicts
    # conflicts[i][j] is a bitset of k such that i, j, k form a sunflower
    # We only need i < j
    conflicts = [[0] * n for _ in range(n)]
    
    # Optimization: Group sets by their core?
    # A sunflower has a core C.
    # If i, j, k form a sunflower with core C, then i&C = j&C = k&C = C.
    # And i\C, j\C, k\C are pairwise disjoint.
    
    # This precomputation might be slow in Python if not careful.
    # 1820^2 / 2 = 1.6M pairs.
    # For each pair, find all k.
    # Current method:
    for i in range(n):
        si = all_sets[i]
        for j in range(i + 1, n):
            sj = all_sets[j]
            core = si & sj
            core_size = bin(core).count('1')
            union = si | sj
            U_mask = ((1 << 16) - 1) ^ union
            d_size = 4 - core_size
            
            if d_size < 0:
                continue
                
            u_elems = [x for x in range(16) if U_mask & (1 << x)]
            
            if d_size > len(u_elems):
                continue
                
            conf_mask = 0
            for d_combo in combinations(u_elems, d_size):
                d_mask = 0
                for x in d_combo:
                    d_mask |= (1 << x)
                sk = core | d_mask
                if sk in mask_to_idx:
                    k = mask_to_idx[sk]
                    if k != i and k != j:
                        conf_mask |= (1 << k)
            
            conflicts[i][j] = conf_mask
            conflicts[j][i] = conf_mask

    # Initial Greedy Solution
    best_family_mask = 0
    best_size = 0
    
    start_time = time.time()
    timeout = 50
    
    # Try a few random greedy starts
    for trial in range(20):
        if time.time() - start_time > timeout:
            break
        random.seed(trial * 12345 + 67890)
        indices = list(range(n))
        random.shuffle(indices)
        
        current_mask = 0
        current_size = 0
        
        for idx in indices:
            # Check if idx can be added
            can_add = True
            cm = current_mask
            while cm:
                u_bit = cm & -cm
                u_idx = u_bit.bit_length() - 1
                cm ^= u_bit
                
                # If idx and u_idx are in the set, do they ban anyone else in the set?
                # No, we just need to check if idx, u_idx, and any other v in current_mask form a sunflower.
                # The condition is: for all v in current_mask, v is NOT in conflicts[idx][u_idx]
                # Actually, if v is in conflicts[idx][u_idx], then idx, u_idx, v form a sunflower.
                # So we must ensure (conflicts[idx][u_idx] & current_mask) == 0
                
                if conflicts[idx][u_idx] & current_mask:
                    can_add = False
                    break
            
            if can_add:
                current_mask |= (1 << idx)
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_family_mask = current_mask

    # Branch and Bound
    # We use the best found so far to prune
    
    def search(candidates_mask, current_mask, current_size):
        nonlocal best_mask, best_size
        
        # Upper bound
        if current_size + candidates_mask.bit_count() <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Reduction: Add all vertices that don't ban any other candidate
        # A vertex v bans u if u is in conflicts[v][w] for some w in current_mask.
        # If v bans no one in candidates, we can add it.
        
        # This check is expensive for all v.
        # Let's just do it for a few or skip if too slow.
        # For now, let's just branch.
        
        # Heuristic: Pick vertex with max ban count
        # To speed up, we pick the first vertex in candidates
        v_bit = candidates_mask & -candidates_mask
        v_idx = v_bit.bit_length() - 1
        
        # Calculate ban mask for v
        ban_mask = 0
        cm = current_mask
        while cm:
            u_bit = cm & -cm
            u_idx = u_bit.bit_length() - 1
            cm ^= u_bit
            ban_mask |= conflicts[v_idx][u_idx]
            
        new_candidates = candidates_mask & ~v_bit & ~ban_mask
        
        # Branch 1: Include v_idx
        search(new_candidates, current_mask | v_bit, current_size + 1)
        
        # Branch 2: Exclude v_idx
        search(candidates_mask & ~v_bit, current_mask, current_size)
        
        if time.time() - start_time > 55:
            raise TimeoutError()

    best_mask = best_family_mask
    
    try:
        all_candidates = (1 << n) - 1
        # Start with empty current set?
        # Or start with the greedy solution?
        # If we start with greedy solution, we need to ensure it's valid (it is).
        # And candidates should be sets that are compatible with the greedy solution.
        
        # Let's start with the greedy solution to get a head start.
        # Candidates = all sets that are compatible with every set in best_family_mask
        # And not in best_family_mask.
        
        candidates = (1 << n) - 1
        cm = best_family_mask
        while cm:
            u_bit = cm & -cm
            u_idx = u_bit.bit_length() - 1
            cm ^= u_bit
            
            # Remove u_idx itself
            candidates &= ~u_bit
            
            # Remove any set that conflicts with u_idx and any other set in best_family_mask
            # This is complex. 
            # Simpler: Just check if a set v can be added to best_family_mask.
            # v is compatible if for all w in best_family_mask, conflicts[v][w] & best_family_mask == 0.
            
            # We can precompute the "compatible" sets for the greedy solution.
            
        # This is getting complicated. Let's just run B&B from scratch with the greedy lower bound.
        # The B&B will find the optimal or better.
        
        search(all_candidates, 0, 0)
        
    except TimeoutError:
        pass
    
    # Extract result
    result = []
    for idx in range(n):
        if best_mask & (1 << idx):
            mask = all_sets[idx]
            elems = []
            for x in range(16):
                if mask & (1 << x):
                    elems.append(x + 1)
            result.append(sorted(elems))
            
    return result
