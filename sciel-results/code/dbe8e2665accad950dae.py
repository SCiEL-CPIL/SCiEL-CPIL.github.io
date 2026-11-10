import random
import time

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    all_masks = list(range(1 << n))
    
    # 1. Deterministic Initialization: Shortlex Initial Segment
    # Sort masks by (bit_count, mask_value)
    def shortlex_key(mask):
        return (bin(mask).count('1'), mask)
    
    sorted_masks = sorted(all_masks, key=shortlex_key)
    family = set(sorted_masks[:m])
    fam_list = list(family)
    
    # Helper to calculate score
    def count_comparable(fam_set):
        count = 0
        fam_list_local = list(fam_set)
        for i in range(len(fam_list_local)):
            a = fam_list_local[i]
            for j in range(i+1, len(fam_list_local)):
                b = fam_list_local[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    current_score = count_comparable(family)
    
    start_time = time.time()
    time_limit = 55.0
    
    # Best-Improvement Local Search with Restarts
    # We will try to find the best single replacement move.
    
    def get_best_move(fam_set, fam_list_local):
        best_delta = 0
        best_a = None
        best_b = None
        
        # To speed up, we can iterate a in fam_list_local
        # and b in all_masks
        # Optimization: Precompute comparable status? No, too heavy.
        
        for a in fam_list_local:
            # For each a, find best b to replace it with
            # Delta = gain(b, F\a) - gain(a, F\a)
            # gain(a, F\a) is constant for this a.
            gain_a = 0
            for f in fam_list_local:
                if f == a:
                    continue
                if (a & f) == a or (a & f) == f:
                    gain_a += 1
            
            # Now find b that maximizes gain(b, F\a)
            # We can break early if we find a very good b?
            # But we need the BEST.
            
            # Optimization: Only check b that are "plausible".
            # But for correctness, check all.
            # To speed up Python loop, use local variables.
            fam_set_local = fam_set
            
            for b in all_masks:
                if b in fam_set_local:
                    continue
                
                gain_b = 0
                for f in fam_list_local:
                    if f == a:
                        continue
                    if (b & f) == b or (b & f) == f:
                        gain_b += 1
                
                delta = gain_b - gain_a
                if delta > best_delta:
                    best_delta = delta
                    best_a = a
                    best_b = b
                    
        return best_delta, best_a, best_b

    # Main Loop
    stuck_count = 0
    while time.time() - start_time < time_limit:
        # Find best move
        best_delta, best_a, best_b = get_best_move(family, fam_list)
        
        if best_delta > 0 and best_a is not None:
            # Apply move
            family.remove(best_a)
            family.add(best_b)
            fam_list[fam_list.index(best_a)] = best_b
            current_score += best_delta
            stuck_count = 0
        else:
            # Stuck in local optimum
            stuck_count += 1
            if stuck_count > 5:
                # Perturbation: Randomly replace 3-5 sets
                num_replacements = random.randint(3, 5)
                for _ in range(num_replacements):
                    idx = random.randint(0, m - 1)
                    old_mask = fam_list[idx]
                    family.remove(old_mask)
                    new_mask = random.choice(all_masks)
                    while new_mask in family:
                        new_mask = random.choice(all_masks)
                    family.add(new_mask)
                    fam_list[idx] = new_mask
                    # Update score (expensive, but done rarely)
                    current_score = count_comparable(family)
                stuck_count = 0

    # Convert to lists of ints
    result = []
    for mask in family:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
