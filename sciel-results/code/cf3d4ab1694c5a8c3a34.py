import math

def construct():
    target = math.factorial(12)
    primes = [2, 3, 5, 7, 11]
    
    # Calculate target exponents
    target_exp = [0] * 5
    n = target
    for i, p in enumerate(primes):
        while n % p == 0:
            target_exp[i] += 1
            n //= p
            
    # Generate all divisors > 12
    divisors = []
    sqrt_t = int(math.isqrt(target))
    for i in range(1, sqrt_t + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            d2 = target // i
            if d2 > 12 and d2 != i:
                divisors.append(d2)
                
    # Remove duplicates and sort
    divisors = sorted(list(set(divisors)))
    
    # Pre-compute exponent vectors for each divisor
    # exp_vecs[i] is the tuple of exponents for divisors[i]
    exp_vecs = []
    for d in divisors:
        vec = [0] * 5
        temp = d
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        exp_vecs.append(tuple(vec))
        
    # We want to find the minimum M such that a subset of divisors <= M 
    # multiplies to target.
    # Iterate M from min(divisors) to max(divisors)
    # Actually, we can just iterate through the sorted divisors as potential max factors.
    
    # To ensure we find the MINIMAL max factor, we try M = divisors[0], divisors[1], etc.
    # But we must ensure that we can form the product using ONLY divisors <= M.
    
    # Optimization: 
    # If we try M, we use the subset of divisors <= M.
    # We can do a binary search on M? No, the property "can form product" is not monotonic in a simple way for subset product?
    # Actually, if we can form it with max factor M1, can we form it with max factor M2 > M1?
    # Yes, because the set of allowed factors grows. But we want the MINIMAL M.
    # So linear scan from smallest M is correct.
    
    # However, the number of divisors is ~360 (since we filter > 12).
    # So we do at most 360 DFS runs.
    # Each DFS run is on a subset of divisors.
    
    # Let's add a stronger pruning:
    # In DFS, we track remaining exponents.
    # We also pre-compute suffix sums of exponents for the current candidate list.
    
    def can_form_product(candidates, target_vec):
        # candidates: list of (divisor, exp_vec)
        # We pick in increasing order of divisor to ensure distinctness.
        # Since candidates are already sorted by divisor, we just pick by index.
        
        n_cand = len(candidates)
        if n_cand == 0:
            return target_vec == (0, 0, 0, 0, 0)
            
        # Precompute suffix sums for pruning
        # suffix_sum[i][j] = sum of exp_vecs[k][j] for k from i to n_cand-1
        suffix_sums = [[0]*5 for _ in range(n_cand + 1)]
        for i in range(n_cand - 1, -1, -1):
            for j in range(5):
                suffix_sums[i][j] = suffix_sums[i+1][j] + candidates[i][1][j]
                
        # Check if total available exponents are sufficient
        for j in range(5):
            if suffix_sums[0][j] < target_vec[j]:
                return None
                
        def dfs(idx, rem_vec):
            # rem_vec: tuple of remaining exponents
            if all(x == 0 for x in rem_vec):
                return []
                
            if idx == n_cand:
                return None
                
            # Pruning: check if remaining candidates can satisfy rem_vec
            for j in range(5):
                if suffix_sums[idx][j] < rem_vec[j]:
                    return None
                    
            # Option 1: Skip candidates[idx]
            res = dfs(idx + 1, rem_vec)
            if res is not None:
                return res
                
            # Option 2: Take candidates[idx]
            d, vec = candidates[idx]
            new_rem = [rem_vec[j] - vec[j] for j in range(5)]
            
            # Check if new_rem has any negative
            if any(x < 0 for x in new_rem):
                return None
                
            new_rem_tuple = tuple(new_rem)
            res = dfs(idx + 1, new_rem_tuple)
            if res is not None:
                return [d] + res
                
            return None
            
        return dfs(0, target_vec)

    # Iterate through possible max factors
    # We iterate through the sorted divisors.
    # For each divisor D, we consider it as the MAX factor.
    # So we filter candidates to be <= D.
    
    # To avoid re-computing the filtered list from scratch every time,
    # we can just incrementally add candidates.
    # But the suffix sums change.
    # Given N~360, recomputing is fine.
    
    for i in range(len(divisors)):
        max_val = divisors[i]
        # Candidates are divisors[0] to divisors[i]
        candidates = [(divisors[j], exp_vecs[j]) for j in range(i + 1)]
        
        sol = can_form_product(candidates, tuple(target_exp))
        
        if sol is not None:
            return sol
            
    # Fallback: return the whole target as a single factor (though it must be > 12, which it is)
    # But the problem asks for DISTINCT integers. [target] is a valid set of 1 distinct integer.
    return [target]
