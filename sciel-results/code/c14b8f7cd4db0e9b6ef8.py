"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    deadline = time.time() + 85.0
    N = math.factorial(12)
    
    # Prime exponents for 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exp = (10, 5, 2, 1, 1)
    primes = [2, 3, 5, 7, 11]
    
    def get_exponents(n):
        exps = [0]*5
        for i, p in enumerate(primes):
            while n % p == 0:
                exps[i] += 1
                n //= p
        return tuple(exps)
    
    # Generate all divisors of 12! that are > 12
    # We represent each factor by its exponent vector
    # We need to find a set of distinct factors (as integers) whose exponent vectors sum to target_exp
    # and all integers are > 12, distinct, and count <= 12
    # Objective: minimize the max factor
    
    # First, let's find the minimum possible max factor via binary search
    # For a given max M, can we partition 12! into distinct factors all in (12, M]?
    
    # Strategy: 
    # 1. Generate all possible factors d where 13 <= d <= M and d divides 12!
    # 2. Use DFS/backtracking to find a subset of these factors (distinct integers) 
    #    whose product is 12! and count <= 12
    # 3. Binary search on M
    
    # To make this efficient, we'll use the exponent vector representation
    # State: remaining exponents needed
    # We pick factors in increasing order to avoid duplicates
    
    def can_partition(max_factor):
        # Get all divisors of 12! in range (12, max_factor]
        # Sort them
        divisors = []
        for d in range(13, max_factor + 1):
            if N % d == 0:
                divisors.append(d)
        
        # Sort divisors
        divisors.sort()
        
        # Precompute exponent vectors
        div_exps = [get_exponents(d) for d in divisors]
        
        # DFS with memoization
        # State: (index, remaining_exp)
        # But remaining_exp is a tuple of 5 ints, which is hashable
        # We also need to ensure we pick at most 12 factors
        
        @lru_cache(maxsize=None)
        def dfs(idx, rem_exp, count):
            if time.time() > deadline:
                return None
            
            if rem_exp == (0, 0, 0, 0, 0):
                if count <= 12:
                    return []
                return None
            
            if count >= 12:
                return None
            
            if idx >= len(divisors):
                return None
            
            # Pruning: can we still reach the target?
            # Check if rem_exp is achievable with remaining divisors
            # Simple check: each component of rem_exp must be >= 0
            for i in range(5):
                if rem_exp[i] < 0:
                    return None
            
            # Option 1: skip current divisor
            result = dfs(idx + 1, rem_exp, count)
            if result is not None:
                return result
            
            # Option 2: take current divisor if it fits
            d_exp = div_exps[idx]
            new_rem = tuple(rem_exp[i] - d_exp[i] for i in range(5))
            
            # Check if new_rem is valid (all >= 0)
            valid = True
            for i in range(5):
                if new_rem[i] < 0:
                    valid = False
                    break
            
            if valid:
                sub_result = dfs(idx + 1, new_rem, count + 1)
                if sub_result is not None:
                    return [divisors[idx]] + sub_result
            
            return None
        
        result = dfs(0, target_exp, 0)
        return result
    
    # Binary search for minimum max_factor
    # Lower bound: 13
    # Upper bound: N (single factor)
    lo = 13
    hi = N
    best_solution = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        sol = can_partition(mid)
        
        if sol is not None:
            best_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_solution is None:
        # Fallback: return a known valid solution
        return [20, 27, 28, 30, 32, 33]
    
    return best_solution
# EVOLVE-BLOCK-END
