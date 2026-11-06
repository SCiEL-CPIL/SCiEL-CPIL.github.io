import sys
from math import factorial
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    timeout = 55.0

    target = factorial(12)
    
    # Get all divisors of target
    all_divs = []
    for i in range(1, int(target ** 0.5) + 1):
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
    
    # Only divisors > 12
    valid_divs = sorted([d for d in all_divs if d > 12])
    n_divs = len(valid_divs)
    
    if n_divs == 0:
        return [target]
        
    # Precompute prefix products of largest k elements for pruning
    # Since candidates change with M (binary search), we compute inside check.
    
    def check_solution(max_idx):
        """
        Checks if a solution exists using valid_divs[0..max_idx].
        Returns the list of factors if found, else None.
        """
        candidates = valid_divs[:max_idx + 1]
        # Sort descending to pick larger factors first, which may prune faster
        candidates = sorted(candidates, reverse=True)
        n = len(candidates)
        
        # Precompute suffix products of largest k elements for pruning
        # max_prod[i][k] = product of largest k elements in candidates[i:]
        # We only need k up to 12.
        # max_prod[i][0] = 1
        # max_prod[i][k] = candidates[i] * max_prod[i+1][k-1] if we take candidates[i]
        # But we want the product of the LARGEST k, so we should always take the largest available.
        # Since candidates is sorted descending, the largest k in candidates[i:] are candidates[i], candidates[i+1], ..., candidates[i+k-1].
        # So max_prod[i][k] = product of candidates[i : i+k].
        
        max_prod = [[1] * 13 for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for k in range(1, 13):
                if i + k <= n:
                    max_prod[i][k] = max_prod[i][k-1] * candidates[i+k-1]
                else:
                    max_prod[i][k] = 0 # Not enough elements
        
        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return True
            if i >= n or count >= 12:
                return False
            
            slots_left = 12 - count
            
            # Pruning:
            # If rem is greater than the product of the largest slots_left available candidates, fail.
            if max_prod[i][slots_left] < rem:
                return False
            
            # Optimization: if only 1 slot left, check if rem is in candidates[i:]
            if slots_left == 1:
                # Since candidates is sorted descending, we can binary search
                # Or just check if rem is in the set of candidates[i:]
                # For speed, let's just rely on the loop, but this check helps avoid deep recursion
                # Actually, the loop will handle it.
                pass
            
            c = candidates[i]
            
            # Option 1: Skip candidates[i]
            if dfs(i + 1, rem, count):
                return True
            
            # Option 2: Take candidates[i]
            if rem % c == 0:
                if dfs(i + 1, rem // c, count + 1):
                    return True
            
            return False

        if dfs(0, target, 0):
            # Reconstruct solution
            result = []
            i, rem, count = 0, target, 0
            while rem != 1:
                if i >= n:
                    break
                c = candidates[i]
                slots_left = 12 - count
                # Check if we can solve by skipping i
                if dfs(i + 1, rem, count):
                    # Skipped
                    i += 1
                elif rem % c == 0 and dfs(i + 1, rem // c, count + 1):
                    # Took
                    result.append(c)
                    rem //= c
                    count += 1
                    i += 1
                else:
                    # Should not happen if dfs returned True
                    break
            
            return result
        else:
            return None

    # Binary search for the minimal M (index in valid_divs)
    low = 0
    high = n_divs - 1
    best_result = None
    
    # Fallback: if binary search fails or times out, return [target]
    # Note: [target] is always a valid solution (1 factor, distinct, >12).
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
        
        mid = (low + high) // 2
        sol = check_solution(mid)
        
        if sol is not None:
            best_result = sol
            high = mid - 1
        else:
            low = mid + 1
        
    if best_result is not None:
        return best_result
    
    # If no solution found in time or via search, return the trivial solution
    # The trivial solution is [12!]
    return [target]

# EVOLVE-BLOCK-END
