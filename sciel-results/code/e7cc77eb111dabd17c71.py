import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target that are > 12
    divisors = set()
    for i in range(13, int(target**0.5) + 1):
        if target % i == 0:
            divisors.add(i)
            other = target // i
            if other > 12:
                divisors.add(other)
    if target > 12:
        divisors.add(target)
    
    D = sorted(divisors)
    n_div = len(D)
    
    # Precompute for each divisor r of target, the indices i in D such that D[i] divides r
    # Note: D[i] must be > 12.
    div_indices = {}
    for r in divisors:
        indices = []
        for i, d in enumerate(D):
            if d > r:
                break
            if r % d == 0:
                indices.append(i)
        div_indices[r] = indices
        
    INF = float('inf')
    min_limit = {}
    
    def get_min_limit(rem, start_idx):
        """
        Returns the minimum index j >= start_idx such that a solution exists
        using divisors[start_idx..j] that multiplies to 'rem'.
        If no such j exists, returns INF.
        """
        if rem == 1:
            # No more factors needed. The max index used is effectively start_idx - 1.
            # We return start_idx - 1 to indicate that no index >= start_idx is needed.
            return start_idx - 1
        
        key = (rem, start_idx)
        if key in min_limit:
            return min_limit[key]
        
        # Pruning
        if start_idx >= n_div:
            min_limit[key] = INF
            return INF
        
        # If the smallest available divisor is > rem, we can't divide.
        if D[start_idx] > rem:
            min_limit[key] = INF
            return INF
        
        best = INF
        # Iterate over indices i such that D[i] divides rem and i >= start_idx
        # We use the precomputed list, but we must filter i >= start_idx
        for i in div_indices[rem]:
            if i < start_idx:
                continue
            # D[i] divides rem
            d = D[i]
            # d <= rem is guaranteed by the construction of div_indices (d > r break)
            
            sub_min = get_min_limit(rem // d, i + 1)
            
            if sub_min != INF:
                # The subproblem is solvable with max index sub_min.
                # We are using index i.
                # Total max index is max(i, sub_min).
                cand = max(i, sub_min)
                if cand < best:
                    best = cand
        
        min_limit[key] = best
        return best
    
    # Find the optimal max index
    min_idx_needed = get_min_limit(target, 0)
    
    if min_idx_needed != INF:
        # Reconstruct the solution
        # We need to find the specific factors.
        # We can use a similar recursive function that picks the path that achieves min_idx_needed.
        
        def reconstruct(rem, start_idx, limit_idx):
            if rem == 1:
                return []
            
            if start_idx > limit_idx:
                return None
            
            # We want to find an i such that:
            # 1. i >= start_idx
            # 2. D[i] divides rem
            # 3. get_min_limit(rem // D[i], i + 1) <= limit_idx
            # 4. max(i, get_min_limit(rem // D[i], i + 1)) == limit_idx (or <= limit_idx, but we want to match the optimal)
            # Actually, we just need ANY solution that fits within limit_idx.
            # Since limit_idx is the minimal possible, any valid solution must use indices <= limit_idx.
            
            for i in div_indices[rem]:
                if i < start_idx:
                    continue
                if i > limit_idx:
                    break # Since div_indices is sorted, we can break
            
            # Wait, div_indices[rem] is sorted.
            # We should iterate and check feasibility.
            
            for i in div_indices[rem]:
                if i < start_idx:
                    continue
                if i > limit_idx:
                    break
                
                d = D[i]
                sub_min = get_min_limit(rem // d, i + 1)
                
                # If sub_min is not INF and sub_min <= limit_idx, then this branch is valid.
                # Also, we need to ensure that the solution can be completed.
                # The condition sub_min <= limit_idx ensures that the subproblem can be solved
                # with indices <= limit_idx.
                
                if sub_min != INF and sub_min <= limit_idx:
                    sub_sol = reconstruct(rem // d, i + 1, limit_idx)
                    if sub_sol is not None:
                        return [d] + sub_sol
            
            return None
        
        sol = reconstruct(target, 0, min_idx_needed)
        if sol is not None:
            return sol
    
    # Fallback
    return [target]
