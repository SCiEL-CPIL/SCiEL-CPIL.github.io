"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import factorial, log, gcd
import time
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin from 90s limit
    
    target = factorial(12)
    
    # Prime factorization of 12!
    # 12! = 479001600
    # = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Let's verify:
    # 2: floor(12/2)+floor(12/4)+floor(12/8) = 6+3+1 = 10
    # 3: floor(12/3)+floor(12/9) = 4+1 = 5
    # 5: floor(12/5) = 2
    # 7: floor(12/7) = 1
    # 11: floor(12/11) = 1
    # Product: 2^10 * 3^5 * 5^2 * 7 * 11 = 1024 * 243 * 25 * 7 * 11
    # = 1024 * 243 = 248832
    # 248832 * 25 = 6220800
    # 6220800 * 7 = 43545600
    # 43545600 * 11 = 479001600 ✓
    
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12!
    def gen_divisors(idx, current_val):
        if idx == len(primes):
            yield current_val
            return
        p = primes[idx]
        e = exponents[idx]
        for k in range(e + 1):
            yield from gen_divisors(idx + 1, current_val * (p ** k))
    
    all_divisors = list(gen_divisors(0, 1))
    valid_divs = sorted([d for d in all_divisors if d > 12])
    
    # Precompute log for pruning
    log_target = log(target)
    
    # Binary search on the maximum factor
    # We want to find the minimum M such that there exists a set of distinct divisors
    # all in [13, M] whose product is target
    
    def can_partition(M):
        """Check if we can partition 12! into distinct factors all <= M, each > 12."""
        candidates = [d for d in valid_divs if d <= M]
        n = len(candidates)
        
        if n == 0:
            return None
        
        # Sort in decreasing order for better pruning
        candidates.sort(reverse=True)
        
        # Precompute suffix product bounds
        suffix_log_prod = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_log_prod[i] = suffix_log_prod[i + 1] + log(candidates[i])
        
        result = []
        
        def backtrack(start, remaining, count):
            if time.time() > deadline:
                return False
            
            if remaining == 1:
                return True
            if count >= 12 or start >= n:
                return False
            
            # Pruning 1: Even taking all remaining candidates, can we reach 'remaining'?
            if suffix_log_prod[start] < log(remaining) - 1e-12:
                return False
            
            # Pruning 2: remaining must be at least 13 (smallest valid factor)
            if remaining < 13:
                return False
            
            # Pruning 3: We need at most (12 - count) more factors
            # The minimum product of (12 - count) factors each >= 13 is 13^(12-count)
            # But we don't need exactly that many, we can use fewer.
            # However, remaining must be divisible by some combination.
            # A stronger bound: if remaining > 1, we need at least one factor.
            # The largest single factor we can pick is candidates[start].
            # If remaining > candidates[start] * (product of all subsequent), we can't do it.
            # That's already covered by suffix_log_prod.
            
            # Additional pruning: if remaining is prime and > M, we can't factor it further
            # But remaining is always a divisor of target, so it's composite or 1.
            
            # Try each candidate
            for i in range(start, n):
                c = candidates[i]
                if c > remaining:
                    continue  # Since sorted descending, all subsequent are smaller, but we still need to check
                if remaining % c != 0:
                    continue
                new_remaining = remaining // c
                result.append(c)
                if backtrack(i + 1, new_remaining, count + 1):
                    return True
                result.pop()
                
                # Pruning: if we skip c, the maximum product from i+1 onwards is suffix_log_prod[i+1]
                # If that's less than log(remaining), we can't achieve remaining without c
                if suffix_log_prod[i + 1] < log(remaining) - 1e-12:
                    break
            
            return False
        
        if backtrack(0, target, 0):
            return result
        return None
    
    # Binary search for the minimum M
    lo = 13
    hi = target  # 12! itself is always a valid single factor
    best_solution = [target]
    best_max = target
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = can_partition(mid)
        if sol is not None:
            best_solution = sol
            best_max = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_solution

# EVOLVE-BLOCK-END
