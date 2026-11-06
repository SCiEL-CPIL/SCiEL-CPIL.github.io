def construct():
    import time
    start = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    from math import factorial
    from functools import lru_cache
    
    N = factorial(12)  # 479001600
    
    # Generate all divisors of N that are > 12
    # Factorize N: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Generate divisors
    divisors = []
    for a in range(11):       # 2^a, a=0..10
        p2 = 2**a
        for b in range(6):    # 3^b, b=0..5
            p3 = p2 * (3**b)
            if p3 > N:
                break
            for c in range(3): # 5^c, c=0..2
                p5 = p3 * (5**c)
                if p5 > N:
                    break
                for d in range(2): # 7^d, d=0..1
                    p7 = p5 * (7**d)
                    if p7 > N:
                        break
                    for e in range(2): # 11^e, e=0..1
                        div = p7 * (11**e)
                        if div > 12 and div < N:  # exclude 1 and N itself? No, N could be a factor if only 1 factor
                            divisors.append(div)
    # Also include N itself as a possible single factor
    divisors.append(N)
    
    # Remove duplicates and sort
    divisors = sorted(set(divisors))
    
    # We want distinct factors > 12, product = N, at most 12 factors
    # Minimize the max factor
    
    best_max = N  # worst case: single factor N
    best_factors = [N]
    
    # DFS: build factors in increasing order
    # state: (remaining_product, min_factor, count, current_factors)
    # We iterate divisors in increasing order, each must be > min_factor (to ensure distinct and sorted)
    
    # Precompute: for quick lookup, which divisors divide a given number
    # Use a set for O(1) lookup
    divisor_set = set(divisors)
    
    # For pruning: given remaining product R and we need at least 1 more factor > min_f,
    # the minimum possible next factor is the smallest divisor of R that is > min_f
    # If no such divisor exists and R != 1, dead end.
    
    # Also: if R itself is a divisor > min_f and R > 12, we can take it as the last factor.
    
    def dfs(remaining, min_factor, count, factors, best_max_ref):
        """
        remaining: product still to be factored
        min_factor: next factor must be > this
        count: number of factors chosen so far
        factors: list of factors chosen so far
        best_max_ref: [best_max] mutable
        Returns: (found, new_best_max) or None
        """
        nonlocal best_max, best_factors
        
        if time.time() > DEADLINE:
            return None
        
        if remaining == 1:
            # All factors chosen
            if count >= 1:
                cur_max = max(factors)
                if cur_max < best_max:
                    best_max = cur_max
                    best_factors = factors[:]
                    return best_max
            return None
        
        if count >= 12:
            return None
        
        # Try taking 'remaining' as the last factor if valid
        if remaining > min_factor and remaining > 12 and remaining in divisor_set:
            cur_max = max(max(factors) if factors else 0, remaining)
            if cur_max < best_max:
                best_max = cur_max
                best_factors = factors + [remaining]
                return best_max
        
        # Try each divisor of remaining that is > min_factor and < remaining
        # Iterate in increasing order
        for d in divisors:
            if d <= min_factor:
                continue
            if d >= remaining:
                break  # since sorted, no more valid
            if remaining % d != 0:
                continue
            # d is a valid next factor
            # Pruning: if d >= best_max, no point (all subsequent are >= d)
            if d >= best_max:
                break
            
            # After taking d, remaining becomes remaining // d
            new_rem = remaining // d
            
            # Check: can new_rem be formed by distinct factors > d?
            # At minimum, if new_rem > 1, we need at least one more factor > d
            # The smallest possible next factor is the smallest divisor of new_rem > d
            # If new_rem == 1, we're done (handled by the "take remaining" case above if we didn't break)
            
            # Quick prune: if new_rem < 2, skip (can't form a factor > d > 12)
            if new_rem <= 1:
                continue
            
            # The next factor must be > d, so the minimum product of remaining factors is at least (d+1)
            # But more precisely, we need new_rem to have a divisor > d
            # Simple check: if new_rem <= d, no divisor > d can divide it (except itself if new_rem > d, but new_rem <= d here)
            if new_rem <= d:
                continue
            
            # Recurse
            result = dfs(new_rem, d, count + 1, factors + [d], best_max_ref)
            if result is not None:
                # best_max may have been updated
                if d >= best_max:
                    break
                # Continue searching for potentially better solutions? 
                # Since we're iterating in increasing order, and we want to minimize max,
                # a smaller d doesn't necessarily mean smaller max. But continuing might find better.
                # However, for efficiency, we can continue.
        
        return None
    
    # Start DFS
    dfs(N, 12, 0, [], [best_max])
    
    return best_factors
