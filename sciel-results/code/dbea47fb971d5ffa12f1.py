import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_one_bitset(values, time_budget=15.0):
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        # Map values to indices 0..20000
        # value = index - 10000
        OFFSET = 10000
        R = 20001
        
        # Precompute masks for each value? No, just use the value directly for shifts.
        # But we need to mask to valid range?
        # Python ints don't have fixed width, so shifts are fine, but we should mask to R bits to avoid overflow issues in logic?
        # Actually, we just care about bits 0..20000.
        # Mask for valid range: (1 << R) - 1
        FULL_MASK = (1 << R) - 1
        
        def val_to_idx(v):
            return v + OFFSET
        
        def idx_to_val(i):
            return i - OFFSET
        
        # Create bitset for all values in V
        V_mask = 0
        for v in values:
            V_mask |= (1 << val_to_idx(v))
        
        # Greedy Add using bitsets
        def can_add(S_mask, v):
            """Check if v can be added to S_mask."""
            idx = val_to_idx(v)
            # 1. v not in S
            if (S_mask >> idx) & 1:
                return False
            
            # 2. 2v not in S
            idx_2v = val_to_idx(2*v)
            if 0 <= idx_2v < R:
                if (S_mask >> idx_2v) & 1:
                    return False
            
            # 3. No y in S such that v+y in S
            # If v > 0: check S & (S >> v)
            # If v < 0: check S & (S << -v)
            if v > 0:
                if S_mask & (S_mask >> v):
                    return False
            elif v < 0:
                if S_mask & (S_mask << (-v)):
                    return False
            
            return True

        def greedy_add(S_list, candidates, S_mask=0):
            result = list(S_list)
            for c in candidates:
                if can_add(S_mask, c):
                    result.append(c)
                    S_mask |= (1 << val_to_idx(c))
            return result, S_mask

        def local_search_bitset(S_list, S_mask, values, time_limit=5.0):
            start = time.time()
            S_set = set(S_list)
            values_set = set(values)
            
            # Phase 1: Add any missing element
            missing = [v for v in values if v not in S_set]
            for v in missing:
                if can_add(S_mask, v):
                    S_list.append(v)
                    S_set.add(v)
                    S_mask |= (1 << val_to_idx(v))
            
            # Phase 2: Local Search (1-remove, multiple-add)
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                # Try removing each element
                for x in list(S_list):
                    if (time.time() - start) >= time_limit:
                        break
                    
                    # Temporarily remove x
                    idx_x = val_to_idx(x)
                    S_mask_test = S_mask & ~(1 << idx_x)
                    S_set_test = S_set - {x}
                    
                    # Try to add any missing element
                    added_any = False
                    for y in values:
                        if y in S_set_test:
                            continue
                        if can_add(S_mask_test, y):
                            S_mask_test |= (1 << val_to_idx(y))
                            S_set_test.add(y)
                            S_list.append(y)
                            added_any = True
                            # Note: We are modifying S_list in place, which is tricky if we revert.
                            # Better to work on a copy or track additions.
                            # For simplicity and speed, let's do a "swap" logic properly.
                            # Actually, the above loop adds ALL possible y.
                            # If len(S_set_test) > len(S_set) - 1, we improved.
                            # But we modified S_list. We need to revert if we don't accept.
                            # Let's use a different structure.
                            pass
                    
                    # This "in-place" modification is dangerous. Let's restart this part.
                    pass
            
            # Let's implement a cleaner local search using copies.
            # Since bitset ops are fast, copying masks is cheap.
            
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                for x in list(S_list):
                    if (time.time() - start) >= time_limit:
                        break
                    
                    idx_x = val_to_idx(x)
                    S_mask_test = S_mask & ~(1 << idx_x)
                    
                    # Try to add all possible elements from V \ {x}
                    # We iterate through V and check can_add
                    new_mask = S_mask_test
                    added_count = 0
                    for y in values:
                        if y == x:
                            continue
                        if (new_mask >> val_to_idx(y)) & 1:
                            continue
                        if can_add(new_mask, y):
                            new_mask |= (1 << val_to_idx(y))
                            added_count += 1
                    
                    # If we added at least 2 elements (net gain of 1), accept
                    if added_count >= 2:
                        # Reconstruct S_list from new_mask
                        S_list = [idx_to_val(i) for i in range(R) if (new_mask >> i) & 1]
                        S_set = set(S_list)
                        S_mask = new_mask
                        improved = True
                        break # Restart loop with new state
            
            return S_list, S_mask

        # Generate initial candidates
        candidates = []
        
        # 1. Odds
        s1 = [x for x in values if x % 2 != 0]
        candidates.append(s1)
        
        # 2. Large Pos
        max_v = max(values)
        thresh = max_v / 2.0
        s2 = [x for x in values if x > thresh]
        candidates.append(s2)
        
        # 3. Large Neg
        min_v = min(values)
        thresh_neg = min_v / 2.0
        s3 = [x for x in values if x < thresh_neg]
        candidates.append(s3)
        
        # 4. Greedy by Abs Value
        s4_list, s4_mask = greedy_add([], sorted(values, key=abs, reverse=True))
        candidates.append(s4_list)
        
        # 5. Random Greedy
        rng = random.Random(42)
        for _ in range(3):
            shuffled = list(values)
            rng.shuffle(shuffled)
            sl, sm = greedy_add([], shuffled)
            candidates.append(sl)

        best_list = []
        best_mask = 0
        best_size = 0
        
        for c in candidates:
            if not is_sum_free(c):
                # Repair
                c_valid = []
                c_set = set()
                for x in c:
                    # Check if x can be added to c_valid
                    # We need to check against c_valid
                    # This is O(N^2) but N is small enough for repair
                    ok = True
                    for y in c_set:
                        if x + y in c_set:
                            ok = False
                            break
                        if y + x in c_set: # same
                            pass
                        if 2*x in c_set:
                            ok = False
                            break
                    if ok:
                        c_valid.append(x)
                        c_set.add(x)
                c = c_valid
            
            if len(c) > best_size:
                best_size = len(c)
                best_list = c
                best_mask = 0
                for x in c:
                    best_mask |= (1 << val_to_idx(x))
        
        # Local Search
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            ts = min(time_budget, remaining_time - 3)
            improved_list, improved_mask = local_search_bitset(best_list, best_mask, values, time_limit=ts)
            if len(improved_list) > best_size:
                best_list = improved_list
                best_mask = improved_mask
                best_size = len(improved_list)
        
        # More restarts if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 8:
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                shuffled = list(values)
                rng.shuffle(shuffled)
                trial_list, trial_mask = greedy_add([], shuffled)
                
                if len(trial_list) >= best_size:
                    ts = min(2.0, deadline - time.time() - 3)
                    if ts > 1:
                        improved_list, improved_mask = local_search_bitset(trial_list, trial_mask, values, time_limit=ts)
                        if len(improved_list) > best_size:
                            best_list = improved_list
                            best_mask = improved_mask
                            best_size = len(improved_list)
        
        return best_list

    results = []
    time_alloc = {60: 12.0, 120: 25.0, 240: 70.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue
        
        values = instance(size, seed)
        budget = time_alloc.get(size, 15.0)
        remaining = deadline - time.time()
        budget = min(budget, max(2.0, remaining - 3.0))
        
        subset = solve_one_bitset(values, time_budget=budget)
        
        if not is_sum_free(subset):
            subset = [x for x in values if x % 2 != 0]
        
        results.append(subset)
    
    return results
