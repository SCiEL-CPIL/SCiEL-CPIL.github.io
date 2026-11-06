import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_one(values, time_budget=15.0):
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        V_set = set(values)
        
        # Map values to indices
        val_to_idx = {v: i for i, v in enumerate(values)}
        n_vals = n
        
        # Precompute conflict structure using bitmasks
        # For each element x, precompute which elements y in V have x+y in V
        # This helps with fast conflict checking
        # conflict_with[i] = bitmask of j such that values[i] + values[j] is in V
        # Also need: for element x, which elements z have x+z in V or z+x in V (same)
        # And 2*x in V
        
        # For fast can_add: given candidate x and current set S (as bitmask),
        # x can be added iff:
        #   For all a in S: a+x not in S and x-a not in S
        #   And 2*x not in S
        # Using bitmask: 
        #   Let mask_S be the bitmask of elements in S
        #   For candidate at index i:
        #   Check: for each j where bit j is set in mask_S:
        #     values[i] + values[j] not in S (i.e., the index of values[i]+values[j] is not set)
        #     values[i] - values[j] not in S
        #   And 2*values[i] not in S
        #
        # Precompute for each pair (i,j): is values[i]+values[j] in V? If so, what index?
        # sum_idx[i][j] = index k if values[i]+values[j] = values[k], else -1
        # diff_idx[i][j] = index k if values[i]-values[j] = values[k], else -1
        
        # For n<=240, we can use Python ints as bitmasks (arbitrary precision)
        # Precompute sum and diff index tables
        sum_idx = [[-1]*n for _ in range(n)]
        diff_idx = [[-1]*n for _ in range(n)]
        
        for i in range(n):
            vi = values[i]
            for j in range(n):
                s = vi + values[j]
                if s in val_to_idx:
                    sum_idx[i][j] = val_to_idx[s]
                d = vi - values[j]
                if d in val_to_idx:
                    diff_idx[i][j] = val_to_idx[d]
        
        # Precompute double index: double_idx[i] = j if 2*values[i] = values[j]
        double_idx = [-1]*n
        for i in range(n):
            d = 2 * values[i]
            if d in val_to_idx:
                double_idx[i] = val_to_idx[d]
        
        def can_add_bitmask(i, mask_S):
            """Check if element at index i can be added to set represented by mask_S."""
            # Check 2*values[i] not in S
            di = double_idx[i]
            if di >= 0 and (mask_S >> di) & 1:
                return False
            # For each j in S, check values[i]+values[j] not in S and values[i]-values[j] not in S
            m = mask_S
            while m:
                # Get lowest set bit
                lb = m & (-m)
                j = lb.bit_length() - 1
                # Check sum
                si = sum_idx[i][j]
                if si >= 0 and (mask_S >> si) & 1:
                    return False
                # Check diff
                di2 = diff_idx[i][j]
                if di2 >= 0 and (mask_S >> di2) & 1:
                    return False
                m ^= lb
            return True
        
        def greedy_add_bitmask(candidates_idx, initial_mask=0):
            """Greedily add candidate indices to initial_mask."""
            mask = initial_mask
            for i in candidates_idx:
                if (mask >> i) & 1:
                    continue
                if can_add_bitmask(i, mask):
                    mask |= (1 << i)
            return mask
        
        def mask_to_list(mask):
            """Convert bitmask to list of values."""
            result = []
            m = mask
            while m:
                lb = m & (-m)
                i = lb.bit_length() - 1
                result.append(values[i])
                m ^= lb
            return result
        
        def list_to_mask(lst):
            """Convert list of values to bitmask."""
            mask = 0
            for v in lst:
                if v in val_to_idx:
                    mask |= (1 << val_to_idx[v])
            return mask
        
        # Generate multiple initial solutions
        candidates = []
        
        # Strategy 1: Odd numbers
        s1 = [x for x in values if x % 2 != 0]
        candidates.append(s1)
        
        # Strategy 2: Large positive numbers (above max/2)
        max_v = max(values)
        thresh_pos = max_v / 2.0
        s2 = [x for x in values if x > thresh_pos]
        candidates.append(s2)
        
        # Strategy 3: Large negative numbers (below min/2)
        min_v = min(values)
        thresh_neg = min_v / 2.0
        s3 = [x for x in values if x < thresh_neg]
        candidates.append(s3)
        
        # Strategy 4: Union of large pos and neg
        s4 = sorted(set(s2).union(s3))
        candidates.append(s4)
        
        # Strategy 5: Greedy by absolute value (largest first)
        s5_idx = sorted(range(n), key=lambda i: abs(values[i]), reverse=True)
        s5_mask = greedy_add_bitmask(s5_idx)
        candidates.append(mask_to_list(s5_mask))
        
        # Strategy 6: Greedy by value (smallest first)
        s6_idx = sorted(range(n), key=lambda i: values[i])
        s6_mask = greedy_add_bitmask(s6_idx)
        candidates.append(mask_to_list(s6_mask))
        
        # Strategy 7: Greedy by value (largest first)
        s7_idx = sorted(range(n), key=lambda i: values[i], reverse=True)
        s7_mask = greedy_add_bitmask(s7_idx)
        candidates.append(mask_to_list(s7_mask))
        
        # Strategy 8: Random shuffles
        rng = Random(42)
        for trial in range(5):
            shuffled_idx = list(range(n))
            rng.shuffle(shuffled_idx)
            sl_mask = greedy_add_bitmask(shuffled_idx)
            candidates.append(mask_to_list(sl_mask))
        
        # Strategy 9: Random perturbation
        for trial in range(3):
            shuffled_idx = list(range(n))
            rng.shuffle(shuffled_idx)
            initial_size = max(1, n // 4)
            initial_mask = 0
            for i in shuffled_idx[:initial_size]:
                if can_add_bitmask(i, initial_mask):
                    initial_mask |= (1 << i)
            sl_mask = greedy_add_bitmask(shuffled_idx[initial_size:], initial_mask)
            candidates.append(mask_to_list(sl_mask))
        
        # Evaluate and keep best
        best = []
        best_mask = 0
        for c in candidates:
            if not is_sum_free(c):
                # Repair by removing conflicting elements
                c_valid = list(c)
                c_set = set(c_valid)
                changed = True
                while changed:
                    changed = False
                    i = 0
                    while i < len(c_valid):
                        x = c_valid[i]
                        conflict = False
                        for y in c_set:
                            if y == x:
                                continue
                            if (x + y) in c_set:
                                conflict = True
                                break
                            if (x - y) in c_set:
                                conflict = True
                                break
                        if conflict:
                            c_valid.pop(i)
                            c_set.remove(x)
                            changed = True
                        else:
                            i += 1
                c = c_valid
            
            if len(c) > len(best):
                best = c
                best_mask = list_to_mask(c)
        
        # Local search with bitmask operations
        def local_search_bitmask(mask, time_limit=5.0):
            start = time.time()
            mask = mask
            
            # Phase 1: Try adding any missing element
            all_mask = (1 << n) - 1
            missing_mask = all_mask ^ mask
            if missing_mask:
                # Try adding each missing element
                m = missing_mask
                while m:
                    lb = m & (-m)
                    i = lb.bit_length() - 1
                    if can_add_bitmask(i, mask):
                        mask |= (1 << i)
                    m ^= lb
            
            # Phase 2: 1-flip moves - remove one, try to add others
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                # Try removing each element and see if we can add more
                m = mask
                while m:
                    if (time.time() - start) >= time_limit:
                        break
                    lb = m & (-m)
                    i = lb.bit_length() - 1
                    m ^= lb
                    
                    # Remove i
                    test_mask = mask & ~(1 << i)
                    
                    # Try adding all missing elements
                    missing = all_mask ^ test_mask
                    mm = missing
                    while mm:
                        lb2 = mm & (-mm)
                        j = lb2.bit_length() - 1
                        if can_add_bitmask(j, test_mask):
                            test_mask |= (1 << j)
                        mm ^= lb2
                    
                    if bin(test_mask).count('1') > bin(mask).count('1'):
                        mask = test_mask
                        improved = True
                        break
            
            # Phase 3: 2-flip moves - remove two, add up to three
            if (time.time() - start) < time_limit:
                improved = True
                while improved and (time.time() - start) < time_limit:
                    improved = False
                    # Get list of indices in mask
                    mask_list = []
                    m = mask
                    while m:
                        lb = m & (-m)
                        i = lb.bit_length() - 1
                        mask_list.append(i)
                        m ^= lb
                    
                    for i in range(len(mask_list)):
                        if (time.time() - start) >= time_limit:
                            break
                        for j in range(i+1, len(mask_list)):
                            if (time.time() - start) >= time_limit:
                                break
                            
                            idx1, idx2 = mask_list[i], mask_list[j]
                            
                            # Remove both
                            test_mask = mask & ~(1 << idx1) & ~(1 << idx2)
                            
                            # Try adding up to 3 missing elements
                            missing = all_mask ^ test_mask
                            mm = missing
                            added = 0
                            while mm and added < 3:
                                lb2 = mm & (-mm)
                                k = lb2.bit_length() - 1
                                if can_add_bitmask(k, test_mask):
                                    test_mask |= (1 << k)
                                    added += 1
                                mm ^= lb2
                            
                            if bin(test_mask).count('1') > bin(mask).count('1'):
                                mask = test_mask
                                improved = True
                                break
                        if improved:
                            break
            
            # Phase 4: Try swap - remove one, add two specific elements
            if (time.time() - start) < time_limit:
                improved = True
                while improved and (time.time() - start) < time_limit:
                    improved = False
                    mask_list = []
                    m = mask
                    while m:
                        lb = m & (-m)
                        i = lb.bit_length() - 1
                        mask_list.append(i)
                        m ^= lb
                    
                    for i in mask_list:
                        if (time.time() - start) >= time_limit:
                            break
                        
                        # Remove i
                        test_mask = mask & ~(1 << i)
                        
                        # Try adding pairs of missing elements
                        missing = all_mask ^ test_mask
                        mm = missing
                        added = 0
                        while mm and added < 2:
                            lb2 = mm & (-mm)
                            j = lb2.bit_length() - 1
                            if can_add_bitmask(j, test_mask):
                                test_mask |= (1 << j)
                                added += 1
                            mm ^= lb2
                        
                        if bin(test_mask).count('1') > bin(mask).count('1'):
                            mask = test_mask
                            improved = True
                            break
            
            return mask
        
        # Local search on best
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            ts = min(time_budget, remaining_time - 3)
            improved_mask = local_search_bitmask(best_mask, time_limit=ts)
            if bin(improved_mask).count('1') > bin(best_mask).count('1'):
                best_mask = improved_mask
                best = mask_to_list(best_mask)
        
        # Try additional random restarts if time permits
        if time.time() < deadline - 10:
            for r in range(15):
                if time.time() > deadline - 5:
                    break
                shuffled_idx = list(range(n))
                rng.shuffle(shuffled_idx)
                sl_mask = greedy_add_bitmask(shuffled_idx)
                # Quick local search
                ts = min(2.5, deadline - time.time() - 2)
                if ts > 0:
                    improved_mask = local_search_bitmask(sl_mask, time_limit=ts)
                    if bin(improved_mask).count('1') > bin(best_mask).count('1'):
                        best_mask = improved_mask
                        best = mask_to_list(best_mask)
        
        return list(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Fallback if time is tight
        if time.time() > deadline - 15:
             results.append([x for x in S if x % 2])
             continue
             
        res = solve_one(S)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to parity
            res = [x for x in S if x % 2]
        
        results.append(res)
        
    return results
