import time
from math import factorial
from functools import lru_cache

def construct():
    target = factorial(12)
    deadline = time.time() + 85
    
    # Get all divisors of target greater than 12
    divisors = []
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i and target // i > 12:
                divisors.append(target // i)
    # Also add target itself if > 12
    if target > 12:
        divisors.append(target)
    
    # Remove duplicates and sort
    divisors = sorted(list(set(divisors)))
    
    best_solution = [target]
    best_max = target
    
    # We want to minimize the largest factor.
    # We iterate M from smallest divisor > 12 upwards.
    # For each M, we check if there is a subset of divisors <= M that multiplies to target.
    
    # To speed up, we can pre-filter divisors <= M.
    # But M changes.
    
    # DFS function
    def dfs(rem, start_idx, current_factors):
        if time.time() > deadline:
            return None
        
        if rem == 1:
            return list(current_factors)
        
        if start_idx >= len(divisors):
            return None
            
        # Pruning: if the smallest available divisor is > rem, we can't form rem (since factors > 1)
        # Wait, if rem > 1, we need a factor <= rem.
        # If divisors[start_idx] > rem, then all subsequent divisors are > rem.
        # So we can return None.
        if divisors[start_idx] > rem:
            return None

        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            
            current_factors.append(d)
            res = dfs(rem // d, i + 1, current_factors)
            if res is not None:
                return res
            current_factors.pop()
            
        return None

    # Iterate M
    # We only need to check M up to best_max.
    # If we find a solution with max factor M, and M < best_max, update best.
    # Since we iterate M upwards, the first solution found for a specific M is not necessarily the global optimal max factor?
    # No, if a solution exists with max factor M, it will be found when we check M.
    # If we check M=100 and find a solution, the max factor is <= 100.
    # But we want to minimize the max factor.
    # So the first M for which a solution exists is the optimal max factor?
    # Yes, because if a solution with max factor M' < M existed, it would have been found when we checked M'.
    # So we can return immediately when we find the first solution?
    # Wait.
    # Suppose a solution exists with factors {20, 30, 40, ...} max 40.
    # When we check M=30, we might not find it because 40 is not in the set.
    # When we check M=40, we find it.
    # So the first M where `dfs` returns non-None is the minimal max factor.
    # So we can return immediately.
    
    # However, we must ensure that the factors found are all <= M.
    # The DFS uses `divisors` which are ALL divisors.
    # We need to restrict the DFS to divisors <= M.
    
    # Let's modify the loop.
    for M in divisors:
        if M >= best_max:
            break
            
        if time.time() > deadline:
            break
            
        # Filter divisors <= M
        # Instead of creating a new list, we can pass a limit index to DFS.
        # Find index of M in divisors.
        # Since divisors is sorted, we can use bisect.
        import bisect
        end_idx = bisect.bisect_right(divisors, M)
        
        # Run DFS with limit end_idx
        # We need a DFS that respects the limit.
        
        def dfs_limited(rem, start_idx, end_limit, current_factors):
            if time.time() > deadline:
                return None
            if rem == 1:
                return list(current_factors)
            if start_idx >= end_limit:
                return None
            if divisors[start_idx] > rem:
                return None
                
            for i in range(start_idx, end_limit):
                d = divisors[i]
                if d > rem:
                    break
                if rem % d != 0:
                    continue
                current_factors.append(d)
                res = dfs_limited(rem // d, i + 1, end_limit, current_factors)
                if res is not None:
                    return res
                current_factors.pop()
            return None

        sol = dfs_limited(target, 0, end_idx, [])
        if sol is not None:
            # This solution uses factors <= M.
            # The max factor is max(sol).
            # Since we iterate M upwards, this is the minimal possible max factor.
            return sol

    return best_solution
