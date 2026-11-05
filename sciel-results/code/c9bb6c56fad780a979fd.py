import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, keep 10s buffer
    
    target = factorial(12)
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    
    divisors = sorted(divisors)
    n_divs = len(divisors)
    
    # Precompute suffix products: suffix_prod[i] = product of divisors[i:]
    # This is huge, so we cap it at target to avoid overflow and speed up comparison
    suffix_prod = [1] * (n_divs + 1)
    for i in range(n_divs - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i + 1] * divisors[i]
        if suffix_prod[i] > target:
            suffix_prod[i] = target + 1  # Cap to avoid huge numbers
    
    # Memoized DFS: returns a tuple of factors or None
    # State: (rem, start_idx, max_idx)
    # We use a dict for memoization since max_idx varies
    
    memo = {}
    
    def can_form(rem, start_idx, max_idx):
        if rem == 1:
            return ()
        
        key = (rem, start_idx, max_idx)
        if key in memo:
            return memo[key]
        
        # Pruning: if product of all available factors from start_idx to max_idx is < rem, fail
        if suffix_prod[start_idx] < rem:
            memo[key] = None
            return None
        
        # Try each factor from start_idx to max_idx
        # Try in ascending order to find solutions that use many small factors
        for i in range(start_idx, max_idx + 1):
            d = divisors[i]
            if rem % d == 0:
                # Check time
                if time.time() > deadline:
                    memo[key] = None
                    return None
                
                res = can_form(rem // d, i + 1, max_idx)
                if res is not None:
                    memo[key] = (d,) + res
                    return memo[key]
        
        memo[key] = None
        return None

    # Iterate M from smallest to largest
    # M is divisors[j]
    for j in range(n_divs):
        M = divisors[j]
        
        # Check if solution exists with max factor <= M
        res = can_form(target, 0, j)
        
        if res is not None:
            # We found a decomposition with max factor <= M.
            # Since we iterate M from small to large, this is the minimal M.
            # The factors in res are distinct and > 12.
            return sorted(res)
        
        # Optional: Check if time is running out
        if time.time() > deadline:
            # Fallback: return [12!]
            return [target]
            
    # If loop finishes without finding (should not happen), return [target]
    return [target]
