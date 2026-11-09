import time
from math import factorial
from functools import lru_cache

_START = time.time()
_DEADLINE = _START + 85

def construct():
    global _START, _DEADLINE
    
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = []
    for a in range(11):  # 2^a
        for b in range(6):  # 3^b
            for c in range(3):  # 5^c
                for d in range(2):  # 7^d
                    for e in range(2):  # 11^e
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append(val)
    
    divisors.sort()
    valid_divs_set = set(divisors)
    
    # We want to minimize the largest factor.
    # Let M be the largest factor.
    # We can binary search for the minimal M.
    # The minimal M must be at least ceil((12!)^(1/1)) = 12! (if 1 factor)
    # But we want to minimize it, so we try smaller M.
    # The maximum possible M is 12! itself.
    # The minimum possible M is bounded by the fact that we need distinct factors > 12.
    # If we use k factors, the max factor is at least (12!)^(1/k).
    # To minimize the max factor, we generally want more factors, but they must be distinct and > 12.
    # Let's just binary search M from a lower bound to an upper bound.
    
    # Lower bound: The smallest possible max factor.
    # If we take the smallest possible distinct factors > 12: 13, 14, 15...
    # Their product grows fast.
    # Let's start lower bound at 13.
    # Upper bound: target (single factor 12! is a valid solution with max 12!).
    
    low = 13
    high = target
    
    best_factors = [target] # Default valid solution
    
    def can_factor(rem, min_allowed, max_allowed, depth):
        """
        Returns a list of factors that multiply to rem, using distinct factors 
        from [min_allowed, max_allowed], or None if impossible.
        Uses memoization.
        """
        if time.time() > _DEADLINE:
            return None
        
        if rem == 1:
            return []
        
        # If rem is prime and rem > max_allowed, we can't factor it further into allowed range
        # Actually, if rem itself is in [min_allowed, max_allowed] and is a divisor, we can take it.
        
        # Try to pick a factor d
        # d must be a divisor of rem
        # d >= min_allowed
        # d <= max_allowed
        # d must be distinct (handled by min_allowed increasing)
        
        # Optimization: 
        # The product of the remaining factors must be rem.
        # If we pick d, the next min_allowed is d+1.
        
        # We iterate d from min_allowed to max_allowed.
        # To be efficient, we should only consider d that are divisors of rem.
        # We can precompute divisors of rem? No, rem changes.
        # But rem is always a divisor of target.
        # We can iterate through the global list of divisors of target that are in [min_allowed, max_allowed] 
        # and check if they divide rem.
        
        # Better: iterate d from min_allowed to max_allowed.
        # Since max_allowed is likely small in the binary search steps where we find the optimum,
        # this loop is short.
        
        # However, if max_allowed is large (e.g., near target), the loop is huge.
        # But we binary search. The "hard" cases are when M is small.
        # When M is small, the range [min_allowed, max_allowed] is small.
        
        # Let's iterate d in the global sorted divisors list, filtered by range.
        
        for d in divisors:
            if d < min_allowed:
                continue
            if d > max_allowed:
                break
            if d > rem:
                break
            
            if rem % d == 0:
                new_rem = rem // d
                # Recurse
                res = can_factor(new_rem, d + 1, max_allowed, depth + 1)
                if res is not None:
                    return [d] + res
        
        return None

    # To speed up, we can use a heuristic: 
    # If we just want to check existence for a given M, we can return bool.
    # But we need the factors to return them.
    # We can first check existence, then reconstruct.
    # Or just return the list.
    
    # Let's use a helper that returns bool for speed, then a second pass for reconstruction if needed.
    # Actually, returning the list is fine if we memoize.
    
    # We need to clear the cache for each new M in binary search.
    
    for M in range(low, high + 1):
        if time.time() > _DEADLINE - 5:
            break
        
        # Clear cache
        can_factor.cache_clear()
        
        # Check if we can factor target using factors <= M
        # Note: factors must be > 12, so min_allowed starts at 13.
        # Also, if M < 13, impossible.
        
        # Optimization: If M is very large, the search might be slow if we just iterate.
        # But we are searching for the MINIMAL M.
        # The minimal M is likely small.
        # Let's estimate: 
        # 12! = 479001600.
        # sqrt(12!) ~ 21886.
        # If we use 2 factors, max is at least 21886.
        # If we use 3 factors, max is at least (12!)^(1/3) ~ 782.
        # If we use 4 factors, max is at least (12!)^(1/4) ~ 267.
        # If we use 5 factors, max is at least (12!)^(1/5) ~ 137.
        # If we use 6 factors, max is at least (12!)^(1/6) ~ 90.
        # If we use 7 factors, max is at least (12!)^(1/7) ~ 69.
        # If we use 8 factors, max is at least (12!)^(1/8) ~ 56.
        # If we use 9 factors, max is at least (12!)^(1/9) ~ 47.
        # If we use 10 factors, max is at least (12!)^(1/10) ~ 41.
        # If we use 11 factors, max is at least (12!)^(1/11) ~ 36.
        # If we use 12 factors, max is at least (12!)^(1/12) ~ 32.
        
        # So the minimal M is likely around 30-100.
        # Iterating M from 13 upwards is very fast.
        
        res = can_factor(target, 13, M, 0)
        if res is not None:
            best_factors = res
            break
    
    return best_factors

