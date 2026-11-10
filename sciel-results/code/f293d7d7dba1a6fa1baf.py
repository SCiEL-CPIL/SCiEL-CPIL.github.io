# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_valid_fast(set_vals):
    """Fast check: no element is the mean of any subset of >=2 other elements."""
    n = len(set_vals)
    if n < 3:
        return True
    arr = np.array(set_vals, dtype=np.int64)
    
    for a in set_vals:
        others = [x for x in set_vals if x != a]
        m = len(others)
        if m < 2:
            continue
        
        # Quick necessary condition: a must be between min and max of others
        # (actually a must be strictly inside the convex hull for it to be a mean)
        if a <= min(others) or a >= max(others):
            continue
        
        max_target = m * a
        if max_target > 50000:
            # Too large for efficient DP, use sampling
            continue
            
        mask = (1 << (max_target + 1)) - 1
        
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
                dp[k] &= mask
            
        for k in range(2, m + 1):
            target = k * a
            if target <= max_target:
                if (dp[k] >> target) & 1:
                    return False
    
    return True

def _check_addition_fast(current_set, x):
    """Check if adding x to current_set creates any violation."""
    new_set = current_set + [x]
    n = len(new_set)
    if n < 3:
        return True
    
    # Check 1: Is x the mean of any subset of >=2 elements from current_set?
    if len(current_set) >= 2:
        others = current_set
        m = len(others)
        
        # Quick check: x must be between min and max
        if x <= min(others) or x >= max(others):
            pass  # Still possible, just skip optimization
        else:
            max_target = m * x
            if max_target <= 50000:
                mask = (1 << (max_target + 1)) - 1
                dp = [0] * (m + 1)
                dp[0] = 1
                for elem in others:
                    for k in range(m, 0, -1):
                        dp[k] |= (dp[k-1] << elem)
                        dp[k] &= mask
                for k in range(2, m + 1):
                    target = k * x
                    if target <= max_target and (dp[k] >> target) & 1:
                        return False
            else:
                # For large targets, check smaller k first
                for k in range(2, min(m+1, 8)):
                    target = k * x
                    # Quick bounds check
                    sorted_others = sorted(others)
                    min_sum = sum(sorted_others[:k])
                    max_sum = sum(sorted_others[-k:])
                    if target < min_sum or target > max_sum:
                        continue
                    # Use recursive check with memoization
                    from functools import lru_cache
                    @lru_cache(maxsize=None)
                    def _exists_subset_sum(elements_tuple, k, target):
                        if k == 0:
                            return target == 0
                        if k > len(elements_tuple):
                            return False
                        if target < 0:
                            return False
                        # Pruning
                        if k > 0:
                            sorted_e = sorted(elements_tuple)
                            min_s = sum(sorted_e[:k])
                            max_s = sum(sorted_e[-k:])
                            if target < min_s or target > max_s:
                                return False
                        for i in range(len(elements_tuple)):
                            if _exists_subset_sum(elements_tuple[:i] + elements_tuple[i+1:], k-1, target - elements_tuple[i]):
                                return True
                        return False
                    if _exists_subset_sum(tuple(others), k, target):
                        return False
                    _exists_subset_sum.cache_clear()
    
    # Check 2: For each a in current_set, is a the mean of some subset of new_set that includes x?
    for a in current_set:
        others_no_a = [y for y in current_set if y != a]
        m = len(others_no_a)
        if m < 1:
            continue
        
        # a = (sum_subset + x) / (t + 1) where t is number of elements from others_no_a
        # So sum_subset = a * (t + 1) - x
        max_target = a * (m + 1) - x
        if max_target < 0:
            continue
        if max_target <= 50000:
            mask = (1 << (max_target + 1)) - 1
            dp = [0] * (m + 1)
            dp[0] = 1
            for elem in others_no_a:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << elem)
                    dp[k] &= mask
            for t in range(1, m + 1):
                target = a * (t + 1) - x
                if 0 <= target <= max_target:
                    if (dp[t] >> target) & 1:
                        return False
        else:
            for t in range(1, min(m+1, 6)):
                target = a * (t + 1) - x
                if target < 0:
                    continue
                # Quick bounds
                sorted_o = sorted(others_no_a)
                min_s = sum(sorted_o[:t])
                max_s = sum(sorted_o[-t:])
                if target < min_s or target > max_s:
                    continue
                from functools import lru_cache
                @lru_cache(maxsize=None)
                def _exists_subset_sum2(elements_tuple, k, target):
                    if k == 0:
                        return target == 0
                    if k > len(elements_tuple):
                        return False
                    if target < 0:
                        return False
                    if k > 0:
                        sorted_e = sorted(elements_tuple)
                        min_s = sum(sorted_e[:k])
                        max_s = sum(sorted_e[-k:])
                        if target < min_s or target > max_s:
                            return False
                    for i in range(len(elements_tuple)):
                        if _exists_subset_sum2(elements_tuple[:i] + elements_tuple[i+1:], k-1, target - elements_tuple[i]):
                            return True
                    return False
                if _exists_subset_sum2(tuple(others_no_a), t, target):
                    return False
                _exists_subset_sum2.cache_clear()
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    # Strategy 1: Greedy from low to high
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_fast(selected, x):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 2: Greedy from high to low
    selected = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_fast(selected, x):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if _check_addition_fast(selected, x):
                selected.append(x)
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Strategy 4: Try to add more elements to best_set
    if time.time() < deadline - 5:
        current = best_set[:]
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in current:
                continue
            if len(current) >= 48:
                break
            if _check_addition_fast(current, x):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Strategy 5: Local search - swap elements
    if time.time() < deadline - 10 and len(best_set) < 48:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for x in range(1, 513):
                    if x in current:
                        continue
                    candidate = current[:i] + [x] + current[i+1:]
                    if _is_valid_fast(candidate):
                        for y in range(1, 513):
                            if y in candidate:
                                continue
                            if len(candidate) >= 48:
                                break
                            if _check_addition_fast(candidate, y):
                                candidate.append(y)
                                improved = True
                                break
                        current = candidate
                        if len(current) > len(best_set):
                            best_set = current[:]
                        break
                if improved:
                    break
    
    # Final: try to fill to 48
    if len(best_set) < 48 and time.time() < deadline - 2:
        current = best_set[:]
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in current:
                continue
            if len(current) >= 48:
                break
            if _check_addition_fast(current, x):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current[:]
    
    return best_set[:48]
# EVOLVE-BLOCK-END
