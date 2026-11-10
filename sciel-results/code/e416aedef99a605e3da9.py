import time
from math import factorial
from functools import lru_cache

TARGET = factorial(12)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Prime factorization of 12!
    # 12! = 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors of TARGET
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    divisors.sort()
    
    # Filter divisors > 12
    # We only care about divisors that are > 12.
    # Also, a divisor d must be such that TARGET/d is an integer (always true).
    # And d must be part of a valid decomposition.
    
    valid_divisors = [d for d in divisors if d > 12]
    
    # We will search for the minimum B (max factor)
    # Iterate B through valid_divisors
    # For each B, candidates are valid_divisors[:i+1] where valid_divisors[i] == B
    
    # Optimization: 
    # If we have a solution with max factor M, we don't need to check B < M.
    # We start from the smallest possible B.
    
    # What is the smallest possible B?
    # If we use 1 factor, B = TARGET.
    # If we use 2 factors, min B is approx sqrt(TARGET) ~ 21886.
    # If we use 3 factors, min B is approx cube root ~ 780.
    # If we use 4 factors, min B is approx 4th root ~ 265.
    # If we use 5 factors, min B is approx 5th root ~ 136.
    # If we use 6 factors, min B is approx 6th root ~ 93.
    # If we use 7 factors, min B is approx 7th root ~ 72.
    # If we use 8 factors, min B is approx 8th root ~ 60.
    # If we use 9 factors, min B is approx 9th root ~ 52.
    # If we use 10 factors, min B is approx 10th root ~ 46.
    # If we use 11 factors, min B is approx 11th root ~ 41.
    # If we use 12 factors, min B is approx 12th root ~ 37.
    
    # However, factors must be distinct and > 12.
    # Let's just iterate B from the smallest valid divisor upwards.
    
    best_solution = None
    best_max_factor = TARGET + 1
    
    # To make the subset search efficient, we should sort candidates.
    # The list valid_divisors is already sorted.
    
    # We will try B values.
    # To avoid checking too many B values, we can jump.
    # But valid_divisors has only ~700 elements. Checking each is fine if the inner search is fast.
    
    for i, B in enumerate(valid_divisors):
        if time.time() > deadline:
            break
            
        # Candidates are valid_divisors[0..i]
        cands = valid_divisors[:i+1]
        n_cands = len(cands)
        
        # Precompute suffix products for pruning
        # suffix_prod[j] = product of cands[j..n_cands-1]
        # If suffix_prod[j] < rem, we can't reach rem using any subset of cands[j..]
        # because even taking all of them isn't enough.
        suffix_prod = [1] * (n_cands + 1)
        for j in range(n_cands - 1, -1, -1):
            suffix_prod[j] = suffix_prod[j+1] * cands[j]
            
        # DFS to find if a subset of cands multiplies to TARGET
        # State: idx, rem, count
        # We use a recursive function with memoization? 
        # rem can be large, but it's a divisor of TARGET.
        # idx is up to 700.
        # count is up to 12.
        
        # We want to find ANY solution, not just one.
        # Actually, if a solution exists for B, it exists for any B' > B.
        # We want the MINIMUM B.
        # So the first B for which DFS returns True is the optimal max factor.
        
        found = False
        
        # To speed up DFS, we can prune if rem is not divisible by any candidate?
        # Better: 
        # 1. If rem == 1: Success (if count >= 1)
        # 2. If count == 12: Fail
        # 3. If idx == n_cands: Fail
        # 4. If suffix_prod[idx] < rem: Fail (not enough mass)
        # 5. If cands[idx] > rem: Fail (all subsequent are larger)
        
        # We also need to ensure we don't pick the same factor twice.
        # Since we iterate idx increasing, and pick cands[idx], distinctness is automatic.
        
        def dfs_subset(idx, rem, count):
            if time.time() > deadline:
                return None
            if rem == 1:
                if count >= 1:
                    return True
                return False
            
            if count >= 12:
                return False
                
            if idx >= n_cands:
                return False
                
            # Prune: not enough product left
            if suffix_prod[idx] < rem:
                return False
                
            # Prune: smallest candidate is larger than rem
            # Since cands is sorted, if cands[idx] > rem, then all cands[j] for j>=idx are > rem.
            # So we can't pick any more.
            if cands[idx] > rem:
                return False
            
            # Option 1: Skip cands[idx]
            if dfs_subset(idx + 1, rem, count):
                return True
                
            # Option 2: Take cands[idx]
            d = cands[idx]
            if rem % d == 0:
                if dfs_subset(idx + 1, rem // d, count + 1):
                    return True
                    
            return False
            
        # We need to reconstruct the solution if found.
        # The boolean DFS above doesn't return the factors.
        # Let's modify it to return the list of factors or None.
        
        def dfs_subset_recon(idx, rem, count, path):
            if time.time() > deadline:
                return None
            if rem == 1:
                if count >= 1:
                    return list(path)
                return None
            
            if count >= 12:
                return None
                
            if idx >= n_cands:
                return None
                
            if suffix_prod[idx] < rem:
                return None
                
            if cands[idx] > rem:
                return None
            
            # Option 1: Skip
            res = dfs_subset_recon(idx + 1, rem, count, path)
            if res is not None:
                return res
                
            # Option 2: Take
            d = cands[idx]
            if rem % d == 0:
                path.append(d)
                res = dfs_subset_recon(idx + 1, rem // d, count + 1, path)
                if res is not None:
                    return res
                path.pop()
                
            return None
            
        solution = dfs_subset_recon(0, TARGET, 0, [])
        
        if solution is not None:
            best_solution = solution
            best_max_factor = B
            # Since we are iterating B in increasing order, the first solution found
            # has the minimum possible maximum factor.
            break

    if best_solution is None:
        return [TARGET]
        
    return best_solution
