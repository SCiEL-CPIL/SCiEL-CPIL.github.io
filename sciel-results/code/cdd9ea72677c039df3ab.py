def construct():
    import time
    import random
    
    deadline = time.time() + 110
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def greedy_sum_free(S, key_func):
        sorted_S = sorted(S, key=key_func)
        result = []
        result_set = set()
        for x in sorted_S:
            ok = True
            for a in result:
                if a + x in result_set:
                    ok = False
                    break
            if ok and 2 * x in result_set:
                ok = False
            if ok:
                result.append(x)
                result_set.add(x)
        return result
    
    def local_search(S, full_set_set, time_budget):
        S_set = set(S)
        full_list = sorted(full_set_set)
        
        def can_add(x, cs):
            for a in cs:
                if a + x in cs:
                    return False
            if 2 * x in cs:
                return False
            return True
        
        start = time.time()
        improved = True
        while improved and (time.time() - start) < time_budget:
            improved = False
            # Try additions
            for x in full_list:
                if time.time() - start > time_budget:
                    break
                if x not in S_set and can_add(x, S_set):
                    S_set.add(x)
                    improved = True
            # Try swaps: for each element in S, try removing it and adding others
            if not improved:
                for a in list(S_set):
                    if time.time() - start > time_budget:
                        break
                    S_set.discard(a)
                    added = 0
                    for x in full_list:
                        if time.time() - start > time_budget:
                            break
                        if x not in S_set and can_add(x, S_set):
                            S_set.add(x)
                            added += 1
                    if added <= 1:
                        S_set.add(a)  # put back
        
        return list(S_set)
    
    def solve_one(full_set, time_budget):
        full_set_list = list(full_set)
        full_set_set = set(full_set_list)
        max_val = max(full_set)
        min_val = min(full_set)
        
        candidates = []
        
        # Large positives
        if max_val > 0:
            candidates.append([x for x in full_set if x > max_val / 2])
        
        # Large negatives
        if min_val < 0:
            candidates.append([x for x in full_set if x < min_val / 2])
        
        # Odds
        candidates.append([x for x in full_set if x % 2 != 0])
        
        # Greedy strategies
        candidates.append(greedy_sum_free(full_set, lambda x: -abs(x)))
        candidates.append(greedy_sum_free(full_set, lambda x: -x))
        candidates.append(greedy_sum_free(full_set, lambda x: x))
        
        # Filter valid
        valid = [c for c in candidates if is_sum_free(c)]
        if not valid:
            return [x for x in full_set if x % 2 != 0]
        
        best = max(valid, key=len)
        best = local_search(best, full_set_set, time_budget)
        return best
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            s = instance(size, seed)
            results.append([x for x in s if x % 2 != 0])
            continue
        full_set = instance(size, seed)
        remaining = deadline - time.time()
        budget = min(remaining * 0.3, 30)
        result = solve_one(full_set, max(1, budget))
        results.append(result)
    
    return results
