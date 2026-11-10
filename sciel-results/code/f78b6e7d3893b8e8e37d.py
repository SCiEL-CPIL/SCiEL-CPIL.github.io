"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache

TARGET = factorial(12)  # 479001600

def _get_all_divisors():
    """Generate all divisors of TARGET, sorted."""
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    divisors.sort()
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # 1. Get all divisors of 12! that are > 12
    all_divs = _get_all_divisors()
    valid_divs = [d for d in all_divs if d > 12]
    n_valid = len(valid_divs)
    
    # 2. Binary search for the smallest maximum factor B
    # The maximum factor must be one of the divisors.
    # We search for the smallest index `hi` in valid_divs such that
    # a subset of valid_divs[0..hi] multiplies to TARGET.
    
    # Feasibility check using DFS + Memoization
    # State: (start_index, remaining_product)
    # We only consider divisors from start_index to hi.
    
    def is_feasible(hi_idx):
        # We need to check if there's a subset of valid_divs[0..hi_idx] 
        # that multiplies to TARGET.
        
        @lru_cache(maxsize=None)
        def dfs(idx, rem):
            # Can we form 'rem' using valid_divs[idx..hi_idx]?
            if rem == 1:
                return True
            if idx > hi_idx:
                return False
            
            # Option 1: Skip valid_divs[idx]
            if dfs(idx + 1, rem):
                return True
            
            # Option 2: Take valid_divs[idx] if it divides rem
            d = valid_divs[idx]
            if rem % d == 0:
                if dfs(idx + 1, rem // d):
                    return True
            
            return False
        
        res = dfs(0, TARGET)
        dfs.cache_clear()
        return res

    # Binary search
    lo = 0
    hi = n_valid - 1
    best_hi = -1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_feasible(mid):
            best_hi = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
        if time.time() > deadline - 5.0:
            break

    if best_hi == -1:
        return [TARGET]

    # 3. Retrieve the factors
    def get_factors(hi_idx):
        @lru_cache(maxsize=None)
        def dfs(idx, rem):
            # Returns a list of factors or None
            if rem == 1:
                return []
            if idx > hi_idx:
                return None
            
            # Option 1: Skip
            res = dfs(idx + 1, rem)
            if res is not None:
                return res
            
            # Option 2: Take
            d = valid_divs[idx]
            if rem % d == 0:
                sub = dfs(idx + 1, rem // d)
                if sub is not None:
                    return [d] + sub
            
            return None
        
        res = dfs(0, TARGET)
        dfs.cache_clear()
        return res

    factors = get_factors(best_hi)
    
    if factors is None:
        return [TARGET]
        
    return factors

# EVOLVE-BLOCK-END
