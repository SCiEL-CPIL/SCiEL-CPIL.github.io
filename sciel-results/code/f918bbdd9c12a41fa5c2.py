import time
import random

def construct():
    start_time = time.time()
    limit = 55.0
    
    N = 512
    candidates = list(range(1, N + 1))
    
    best_set = []
    
    def has_subset_sum_bitset(vals, target, min_size):
        if min_size > len(vals):
            return False
        if min_size == 1:
            return target in vals
            
        max_abs_sum = sum(abs(v) for v in vals)
        if max_abs_sum == 0:
            return target == 0 and min_size <= len(vals)
            
        offset = max_abs_sum
        max_size = len(vals)
        dp = [0] * (max_size + 1)
        dp[0] = 1 << offset
        
        for v in vals:
            for k in range(max_size - 1, -1, -1):
                if v >= 0:
                    dp[k+1] |= dp[k] << v
                else:
                    dp[k+1] |= dp[k] >> -v
                    
        target_idx = target + offset
        for k in range(min_size, max_size + 1):
            if (dp[k] >> target_idx) & 1:
                return True
        return False

    def can_add(x, S):
        if not S:
            return True
            
        vals1 = [v - x for v in S]
        if has_subset_sum_bitset(vals1, 0, 2):
            return False
            
        for y in S:
            vals2 = [v - y for v in S if v != y]
            target = y - x
            if has_subset_sum_bitset(vals2, target, 1):
                return False
                
        return True

    def backtrack(start_idx, current_set):
        nonlocal best_set
        if time.time() - start_time > limit:
            return
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        for i in range(start_idx, N):
            if time.time() - start_time > limit:
                return
            x = candidates[i]
            if can_add(x, current_set):
                current_set.append(x)
                backtrack(i + 1, current_set)
                current_set.pop()
                
    backtrack(0, [])
    
    if time.time() - start_time < limit - 5:
        for _ in range(50):
            if time.time() - start_time > limit - 2:
                break
            random.shuffle(candidates)
            backtrack(0, [])
            
    return best_set
