"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    n = 12
    target = factorial(n)  # 479001600

    # We need distinct integers > n whose product is exactly target.
    # Minimize the largest factor.
    
    # Strategy:
    # 1. Generate all divisors of target that are > n.
    # 2. Sort them.
    # 3. Use a search (backtracking / DFS) to find a subset of distinct divisors
    #    whose product is target, minimizing the maximum element in the subset.
    #    Since we want to minimize the max, we can binary search on the max value M,
    #    and check if there exists a subset of distinct divisors (all <= M, all > n)
    #    whose product is target.
    
    # First, generate all divisors of target.
    divisors = set()
    i = 1
    while i * i <= target:
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
        i += 1
    
    # Filter: must be > n
    divisors = [d for d in divisors if d > n]
    divisors.sort()
    
    # We want to find a subset of these divisors (distinct, by value) such that
    # their product is exactly target, and we minimize the maximum value in the subset.
    # Also, the number of factors must be between 1 and 12.
    
    # Note: Since the factors must be distinct integers > n, and their product is target,
    # and target is fixed, the set of possible factors is bounded.
    
    # Binary search on M (the maximum allowed factor value).
    # For a given M, we want to know if there's a subset of divisors (all <= M, all > n)
    # with product exactly target.
    
    # However, a direct subset product search is NP-hard in general, but here the
    # number of divisors is manageable. Let's count them.
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Number of divisors = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792
    # Divisors > 12: slightly fewer, still ~700-800.
    
    # A DFS over 800 items is too slow for subset product.
    # But we can use the prime exponent vector representation.
    
    # Let's represent each divisor by its exponent vector (e2, e3, e5, e7, e11).
    # Target vector: (10, 5, 2, 1, 1)
    
    primes = [2, 3, 5, 7, 11]
    
    def factorize(x):
        vec = [0] * 5
        for i, p in enumerate(primes):
            while x % p == 0:
                vec[i] += 1
                x //= p
        return tuple(vec)
    
    target_vec = tuple(factorize(target))
    
    # Build list of (value, vec) for divisors > n
    candidates = []
    for d in divisors:
        v = factorize(d)
        candidates.append((d, v))
    
    # Sort by value
    candidates.sort(key=lambda x: x[0])
    
    # Now, for a given M, we consider only candidates with value <= M.
    # We need to find a subset with product vector == target_vec.
    
    # This is a multi-dimensional subset sum problem.
    # Given the small number of dimensions (5) and the bounded exponents,
    # we can use a BFS/DFS with memoization on the remaining vector.
    
    # But the state space is (11*6*3*2*2) = 792 states.
    # For each state, we try to add a candidate.
    # This is feasible if we do it smartly.
    
    # Let's try a different approach: since we want to minimize M, and M must be
    # one of the divisor values, we can iterate M in increasing order and check
    # feasibility.
    
    # For feasibility check with max value M:
    # Use DFS with memoization. State: (current_vec, index_in_candidates)
    # But index_in_candidates can be up to ~800, and current_vec has 792 values.
    # Total states: 800 * 792 ≈ 632,000. This is feasible.
    
    # Actually, we can do better: since we're adding candidates in increasing order
    # to ensure distinctness, we can use a recursive DFS that picks the next candidate
    # from a sorted list, ensuring we don't reuse.
    
    # Let's implement a feasibility check:
    # Given a list of candidates (value, vec) with value <= M,
    # is there a subset whose vec sum equals target_vec?
    
    # We'll use a set of reachable vectors, starting from (0,0,0,0,0).
    # For each candidate, we update the set of reachable vectors.
    # But this is O(n * |states|) which is 800 * 792 ≈ 632,000 operations per M.
    # And we might check many M values. This could be slow.
    
    # Alternative: since the number of factors is at most 12, we can do a DFS
    # that builds the subset, pruning when the product exceeds target or
    # the vector exceeds target_vec.
    
    # Let's try a DFS approach with pruning:
    
    def can_form_subset(candidates, target_vec, max_factors=12):
        """
        candidates: list of (value, vec) sorted by value.
        Returns True if a subset exists with vec sum == target_vec and len <= max_factors.
        """
        n_cand = len(candidates)
        
        # Use memoization: (index, current_vec) -> bool
        # But current_vec is a tuple of 5 ints, each bounded.
        # We'll use a set for visited states to avoid re-exploration.
        
        from functools import lru_cache
        
        # Convert target_vec to a comparable tuple
        @lru_cache(maxsize=None)
        def dfs(idx, e2, e3, e5, e7, e11, count):
            if count > max_factors:
                return False
            if (e2, e3, e5, e7, e11) == target_vec:
                return True
            if idx >= n_cand:
                return False
            # Prune: if any exponent exceeds target, return False
            if e2 > target_vec[0] or e3 > target_vec[1] or e5 > target_vec[2] or e7 > target_vec[3] or e11 > target_vec[4]:
                return False
            
            # Option 1: skip candidate[idx]
            if dfs(idx + 1, e2, e3, e5, e7, e11, count):
                return True
            
            # Option 2: take candidate[idx]
            v = candidates[idx][1]
            new_e2 = e2 + v[0]
            new_e3 = e3 + v[1]
            new_e5 = e5 + v[2]
            new_e7 = e7 + v[3]
            new_e11 = e11 + v[4]
            
            if new_e2 <= target_vec[0] and new_e3 <= target_vec[1] and new_e5 <= target_vec[2] and new_e7 <= target_vec[3] and new_e11 <= target_vec[4]:
                if dfs(idx + 1, new_e2, new_e3, new_e5, new_e7, new_e11, count + 1):
                    return True
            
            return False
        
        return dfs(0, 0, 0, 0, 0, 0)
    
    # Now binary search on M.
    # M must be at least the smallest possible max factor.
    # Lower bound: if we use k factors, the max must be at least (target)^(1/k).
    # But we don't know k. Let's just iterate M over the candidate values.
    
    # Since we want to minimize M, let's try M values in increasing order.
    # But there are ~800 candidates. For each, the DFS might be expensive.
    
    # Let's optimize: use the fact that if M is small, the number of candidates
    # with value <= M is small.
    
    best_M = None
    best_subset = None
    
    # Iterate M in increasing order
    for i, (val, vec) in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Consider all candidates with value <= val
        sub_candidates = candidates[:i+1]
        
        if can_form_subset(sub_candidates, target_vec):
            best_M = val
            # Now find the actual subset
            # We need to modify the DFS to also return the subset
            break
    
    if best_M is None:
        # Fallback: return [target]
        return [target]
    
    # Now find the actual subset that achieves product = target with max factor = best_M
    # Use a similar DFS but this time record the subset
    
    def find_subset(candidates, target_vec, max_factors=12):
        n_cand = len(candidates)
        result = []
        
        def dfs(idx, e2, e3, e5, e7, e11, count, current_subset):
            if count > max_factors:
                return False
            if (e2, e3, e5, e7, e11) == target_vec:
                result.extend(current_subset)
                return True
            if idx >= n_cand:
                return False
            if e2 > target_vec[0] or e3 > target_vec[1] or e5 > target_vec[2] or e7 > target_vec[3] or e11 > target_vec[4]:
                return False
            
            # Option 1: skip
            if dfs(idx + 1, e2, e3, e5, e7, e11, count, current_subset):
                return True
            
            # Option 2: take
            v = candidates[idx][1]
            new_e2 = e2 + v[0]
            new_e3 = e3 + v[1]
            new_e5 = e5 + v[2]
            new_e7 = e7 + v[3]
            new_e11 = e11 + v[4]
            
            if new_e2 <= target_vec[0] and new_e3 <= target_vec[1] and new_e5 <= target_vec[2] and new_e7 <= target_vec[3] and new_e11 <= target_vec[4]:
                if dfs(idx + 1, new_e2, new_e3, new_e5, new_e7, new_e11, count + 1, current_subset + [candidates[idx][0]]):
                    return True
            
            return False
        
        dfs(0, 0, 0, 0, 0, 0, result)
        return result
    
    sub_candidates = [c for c in candidates if c[0] <= best_M]
    subset = find_subset(sub_candidates, target_vec)
    
    if not subset:
        return [target]
    
    # Verify
    prod = 1
    for x in subset:
        prod *= x
    if prod != target:
        return [target]
    
    if len(set(subset)) != len(subset):
        return [target]
    
    if any(x <= n for x in subset):
        return [target]
    
    return subset

# EVOLVE-BLOCK-END
