import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exponents = (10, 5, 2, 1, 1)
    primes = [2, 3, 5, 7, 11]
    
    # Generate all divisors of N that are > 12
    divisors = []
    for i in range(13, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            other = N // i
            if other != i and other > 12:
                divisors.append(other)
    # Add N itself? No, we want distinct factors, and if we use N, we can't use others.
    # But N is a valid factor if k=1.
    # However, we want to minimize the largest factor.
    # If we use N, largest is N.
    # If we use smaller factors, largest is smaller.
    # So we should consider all divisors.
    # The loop above finds divisors up to sqrt(N).
    # We need to ensure we have all divisors > 12.
    # Let's generate all divisors properly.
    
    divisors = set()
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            if i > 12:
                divisors.add(i)
            other = N // i
            if other > 12:
                divisors.add(other)
                
    sorted_divisors = sorted(divisors)
    
    # Map integer to its exponent vector
    def get_exponents(n):
        exps = [0, 0, 0, 0, 0]
        for i, p in enumerate(primes):
            while n % p == 0:
                exps[i] += 1
                n //= p
        return tuple(exps)
        
    factor_vectors = {}
    for d in sorted_divisors:
        factor_vectors[d] = get_exponents(d)
        
    # We want to find a subset of distinct factors from {d | d <= M}
    # such that sum of their exponent vectors == target_exponents.
    
    # DP State: (idx, e2, e3, e5, e7, e11)
    # idx: index in the list of available factors (sorted by value)
    # We iterate M from 13 upwards.
    
    # To speed up, we can precompute the list of factors for each M?
    # Or just filter on the fly.
    
    # Since the number of factors is small (~100-200), we can use a DFS with memoization
    # that takes the set of available factors as a bitmask?
    # No, the set changes with M.
    
    # Better:
    # Iterate M from 13 to max(sorted_divisors).
    # For each M, let candidates = [d for d in sorted_divisors if d <= M].
    # Run DP on candidates.
    
    # DP function
    def solve(candidates):
        # candidates is a list of (value, vector)
        n = len(candidates)
        if n == 0:
            return None
            
        # Memoization cache
        # Key: (index, e2, e3, e5, e7, e11)
        # Value: tuple of chosen values or None
        memo = {}
        
        def dfs(idx, e2, e3, e5, e7, e11, path):
            if e2 == 0 and e3 == 0 and e5 == 0 and e7 == 0 and e11 == 0:
                return path
                
            if idx == n:
                return None
                
            if (idx, e2, e3, e5, e7, e11) in memo:
                return memo[(idx, e2, e3, e5, e7, e11)]
                
            # Option 1: Skip candidates[idx]
            res = dfs(idx + 1, e2, e3, e5, e7, e11, path)
            if res:
                memo[(idx, e2, e3, e5, e7, e11)] = res
                return res
                
            # Option 2: Take candidates[idx]
            val, vec = candidates[idx]
            ne2 = e2 - vec[0]
            ne3 = e3 - vec[1]
            ne5 = e5 - vec[2]
            ne7 = e7 - vec[3]
            ne11 = e11 - vec[4]
            
            if ne2 >= 0 and ne3 >= 0 and ne5 >= 0 and ne7 >= 0 and ne11 >= 0:
                path.append(val)
                res = dfs(idx + 1, ne2, ne3, ne5, ne7, ne11, path)
                if res:
                    memo[(idx, e2, e3, e5, e7, e11)] = res
                    return res
                path.pop()
                
            memo[(idx, e2, e3, e5, e7, e11)] = None
            return None
            
        return dfs(0, target_exponents[0], target_exponents[1], target_exponents[2], target_exponents[3], target_exponents[4], [])

    for M in range(13, 1000):
        if time.time() - start_time > 85:
            break
            
        candidates = [(d, factor_vectors[d]) for d in sorted_divisors if d <= M]
        if not candidates:
            continue
            
        sol = solve(candidates)
        if sol:
            return sol
            
    return [N]
