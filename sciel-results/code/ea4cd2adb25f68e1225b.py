def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 16
    k = 4
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for c in combinations(range(n), k):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
        
    num_sets = len(all_masks)
    best_family = []
    
    def is_compatible(x, S_masks):
        # S_masks is list of ints
        groups = {}
        for a in S_masks:
            inter = a & x
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(a)
            
        for K, group in groups.items():
            if len(group) < 2:
                continue
            # Check pairs
            for i in range(len(group)):
                a = group[i]
                a_tail = a & ~K # Elements in a not in K
                for j in range(i+1, len(group)):
                    b = group[j]
                    b_tail = b & ~K
                    if (a_tail & b_tail) == 0:
                        return False
        return True

    import random
    max_trials = 5000
    trial = 0
    while trial < max_trials:
        if time.time() > deadline:
            break
        
        random.shuffle(all_masks)
        S = []
        for x in all_masks:
            if is_compatible(x, S):
                S.append(x)
        
        if len(S) > len(best_family):
            best_family = S
            # Optional: if best_family is large, we can try to improve it
            # with local search, but let's see if greedy is enough
            
        trial += 1
        
    # Convert best_family masks back to sorted lists
    result = []
    for mask in best_family:
        elems = [i+1 for i in range(n) if (mask >> i) & 1]
        result.append(elems)
        
    return result
