import math
from functools import lru_cache

def construct():
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter divisors > 12
    divisors = [d for d in divisors if d > 12]
    divisors.sort()
    
    # We want to minimize the largest factor.
    # Iterate M from smallest possible divisor to largest.
    # For each M, check if a subset of divisors <= M exists.
    
    # To speed up, we can just sort divisors and use binary search for M?
    # No, just iterate through unique values of divisors as potential M.
    # Actually, M must be one of the divisors in the solution.
    # So we can iterate through sorted divisors as potential max.
    
    # However, it's possible the max factor is a divisor, but we need to check subsets of ALL divisors <= M.
    # So:
    # for M in sorted_divisors:
    #    candidates = [d for d in sorted_divisors if d <= M]
    #    if search(candidates, N):
    #        return solution
    
    # Optimization: 
    # The number of divisors is small (720).
    # The inner search is a subset product problem.
    # We can use a set of reachable products? 
    # No, we need the factors themselves.
    
    # Backtracking search function
    def find_subset(candidates, target):
        # candidates is sorted list of divisors <= M
        # We need distinct elements from candidates that multiply to target
        # Since candidates are sorted, we can use index to ensure distinctness
        
        n = len(candidates)
        # Pruning: if product of all candidates < target, impossible
        # if product of largest k candidates < target, impossible
        
        # Simple DFS
        # To avoid deep recursion issues or timeouts, we limit depth to 12
        # Also, we can sort candidates and use two-pointer style or just standard DFS
        
        # Let's use a helper that returns the list of factors
        def dfs(start_idx, current_prod, chosen):
            if current_prod == target:
                if 1 <= len(chosen) <= 12:
                    return chosen
                else:
                    return None
            
            if current_prod > target:
                return None
            
            # Pruning: 
            # If we multiply current_prod by the largest possible remaining product, 
            # can we reach target?
            # Max possible product from remaining candidates (largest ones)
            # This is hard to compute quickly. 
            # But since target is fixed and we are in early stages of M, 
            # the search space is small.
            
            for i in range(start_idx, n):
                d = candidates[i]
                # If current_prod * d > target, break (since sorted)
                if current_prod * d > target:
                    break
                
                # Additional pruning: 
                # If we pick d, the remaining product needed is target // current_prod // d
                # We must be able to form this with elements > d (indices > i)
                
                res = dfs(i + 1, current_prod * d, chosen + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(0, 1, [])

    # Iterate through possible maximum values
    # We only need to check M values that are in divisors
    for M in divisors:
        # Get candidates <= M
        # Since divisors is sorted, we can find the index
        # But we are iterating M from divisors, so we can just slice
        # Wait, divisors list is sorted. 
        # We can maintain a pointer or just filter. 
        # Filtering 720 items is fast.
        
        # Actually, we can just use the list of divisors up to M.
        # Since we iterate M in sorted order, we can build the candidate list incrementally?
        # No, the DFS depends on the specific set.
        
        # Let's just filter.
        candidates = [d for d in divisors if d <= M]
        
        # Quick check: is it possible?
        # Product of all candidates must be >= target?
        # If product of all candidates < target, skip this M.
        prod_all = 1
        for d in candidates:
            prod_all *= d
            if prod_all > N:
                break
        if prod_all < N:
            continue
            
        sol = find_subset(candidates, N)
        if sol:
            return sol
            
    # Fallback: return the single factor
    return [N]
