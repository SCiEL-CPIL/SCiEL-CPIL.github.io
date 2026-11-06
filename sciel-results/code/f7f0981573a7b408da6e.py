import sys
from math import factorial
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    timeout = 55.0

    target = factorial(12)
    
    # Get all divisors of target > 12
    all_divs = []
    for i in range(1, int(target ** 0.5) + 1):
        if target % i == 0:
            if i > 12:
                all_divs.append(i)
            j = target // i
            if j != i and j > 12:
                all_divs.append(j)
    
    valid_divs = sorted(all_divs)
    n_divs = len(valid_divs)
    
    if n_divs == 0:
        return [target]

    def check_solution(max_idx):
        candidates = valid_divs[:max_idx + 1]
        n = len(candidates)
        
        # Precompute product of largest k elements in candidates
        prod_largest = [1] * 13
        for k in range(1, 13):
            if n >= k:
                p = 1
                for j in range(n-k, n):
                    p *= candidates[j]
                prod_largest[k] = p
            else:
                prod_largest[k] = 0
                
        # Precompute suffix products for "not enough elements" case
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]

        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return True
            if i >= n or count >= 12:
                return False
            
            slots_left = 12 - count
            
            # Pruning
            if n - slots_left >= i:
                # We have enough elements to pick slots_left
                # Max product is product of largest slots_left elements
                if prod_largest[slots_left] < rem:
                    return False
            else:
                # Not enough elements to fill slots_left
                # Max product is product of all remaining elements
                if suffix_prod[i] < rem:
                    return False

            # Skip
            if dfs(i + 1, rem, count):
                return True
            
            # Take
            c = candidates[i]
            if rem % c == 0:
                if dfs(i + 1, rem // c, count + 1):
                    return True
            
            return False

        if dfs(0, target, 0):
            # Reconstruct
            result = []
            i, rem, count = 0, target, 0
            while rem != 1:
                if i >= n:
                    break
                c = candidates[i]
                if dfs(i + 1, rem, count):
                    i += 1
                elif rem % c == 0 and dfs(i + 1, rem // c, count + 1):
                    result.append(c)
                    rem //= c
                    count += 1
                    i += 1
                else:
                    break
            return result
        else:
            return None

    # Binary search for minimal max_idx
    low = 0
    high = n_divs - 1
    best_result = None
    
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
        
    return [target]
