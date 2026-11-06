import math
import time

def construct():
    start_time = time.time()
    time_limit = 145 # Safety margin
    
    max_val = 10**18
    # Primes up to 71. 71 is included because 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*71 
    # might be within limits if exponents are low, but typically 47 is the limit for squarefree.
    # However, for non-squarefree, we might use larger primes with exponent 1 while skipping smaller ones?
    # No, the DFS enforces non-increasing exponents. So if we use 71 with e=1, all previous must have e>=1.
    # The product of primes up to 71 is huge.
    # Let's stick to primes up to 47 for the main search, but maybe check 53, 59, 61, 67, 71 if the product allows.
    # Product of first 15 primes (up to 47) is ~6.1e17.
    # Product of first 16 primes (up to 53) is ~3.2e19 > 10^18.
    # So squarefree numbers can have at most 15 distinct prime factors.
    # If we use 53, we must drop some smaller prime.
    # The DFS with non-increasing exponents naturally handles this by exploring combinations.
    # We should include primes up to maybe 100 to be safe, but the product constraint is tight.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    candidates = [] # List of (tau, n)
    
    # DFS to generate candidates with non-increasing exponents
    # Optimization: Prune branches where even with max possible tau from remaining primes, we can't beat current best?
    # Hard to know current best early.
    # Instead, just generate all valid n with non-increasing exponents.
    # The number of such n is manageable.
    
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > 100: # Stop generating after 100s
            return
            
        if current_n > 1:
            candidates.append((current_tau, current_n))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        # Calculate max exponent for this prime
        e_max = 0
        temp = 1
        # Ensure e <= max_exp
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1
            
        # If e_max is 0, we can skip this prime and all subsequent?
        # No, we can skip this prime and try the next one with the same max_exp.
        # But the non-increasing constraint applies to the sequence of exponents used.
        # If we skip p, the next prime p' must have exponent <= current max_exp.
        # Wait, the standard "highly composite" style generation assumes we use primes in order.
        # If we skip a prime, we are effectively setting its exponent to 0.
        # The constraint is e_1 >= e_2 >= ... >= e_k > 0.
        # If e_i = 0, then e_{i+1} must be 0? No, that's not right.
        # The set of primes used is a subset.
        # Let's say we use primes p_{i1}, p_{i2}, ... with exponents e_1 >= e_2 >= ...
        # The current DFS structure `dfs(idx, max_exp, ...)` implies we are deciding the exponent for primes[idx].
        # If we set exponent to 0, we should proceed to idx+1 with the SAME max_exp?
        # Yes.
        
        # Option 1: Skip this prime (exponent 0)
        dfs(idx + 1, max_exp, current_n, current_tau)
        
        # Option 2: Use this prime with exponent e from 1 to e_max
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > 100:
                return

    # Start DFS
    # Max exponent for 2 is ~59
    dfs(0, 60, 1, 1)
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    # Sort by tau descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate? The DFS with skip option might generate duplicates?
    # No, each path corresponds to a unique assignment of exponents to the first k primes (where k is the max index reached).
    # Wait, if we skip 2, then use 3, that's a unique path.
    # If we use 2^1, skip 3, use 5, that's unique.
    # So no duplicates in n?
    # Yes, the mapping from (e_0, e_1, ..., e_{k-1}) to n is unique.
    
    def tau_perp(n):
        # Factorize n using small primes
        factors = []
        temp_n = n
        for p in primes:
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                factors.append((p, count))
            if temp_n == 1:
                break
        
        if temp_n > 1:
            # Should not happen if primes list is sufficient, but just in case
            # temp_n is a prime > 97
            factors.append((temp_n, 1))

        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
                    
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_score = -1
    best_n = candidates[0][1]
    
    # Evaluate top candidates
    # We can evaluate more if we are fast
    # Heuristic: The best score is likely among the top 1000 by tau.
    # But let's evaluate as many as possible within time.
    
    for tau, n in candidates:
        if time.time() - start_time > time_limit - 5:
            break
            
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
        # Early stopping heuristic?
        # If tau is significantly lower than the best tau seen, and we've seen many candidates, 
        # maybe stop? 
        # But tau_perp is not strictly monotonic with tau.
        # However, higher tau generally means more divisors, more pairs.
        # Let's just go through all candidates if time permits.
        
    return best_n
