"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    target = math.factorial(12)
    deadline = time.time() + 85.0
    
    # Generate all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
            
    # Filter divisors > 12
    candidates = sorted([d for d in divisors if d > 12])
    
    # We want to minimize the largest factor.
    # We iterate through potential maximum factors K (from candidates) in increasing order.
    # For each K, we check if there is a subset of distinct candidates <= K that multiplies to target.
    # The first K for which this is true gives the optimal solution.
    
    best_factors = [target] # Fallback
    
    for k_idx, K in enumerate(candidates):
        # Candidates allowed are candidates[0..k_idx]
        # We need to find a subset of these that multiplies to target.
        
        # Optimization: If target is not divisible by any product of a subset, skip.
        # We use DFS to find a subset.
        # State: remaining, start_idx
        # We want to know if a solution exists.
        
        def exists_subset(remaining, start_idx, k_idx_limit):
            if remaining == 1:
                return True
            
            if start_idx > k_idx_limit:
                return False
                
            # Pruning: if remaining is not divisible by any candidate in candidates[start_idx..k_idx_limit], return False?
            # This check is expensive. 
            # Instead, we can just recurse.
            
            # Heuristic: If remaining is prime, it must be one of the candidates.
            # But checking primality is also overhead.
            
            # Simple bound: The product of the smallest (k_idx_limit - start_idx + 1) candidates must be <= remaining?
            # No, we can pick any subset.
            
            # If remaining < candidates[start_idx], we can't pick any more factors (since all are > 12 and distinct, and we are picking in increasing order? No, we are picking from a set).
            # Actually, if we pick factors in increasing index order, the values are increasing.
            # If remaining < candidates[start_idx], we can't form remaining with any subset of candidates[start_idx..] because the smallest factor is candidates[start_idx] > remaining.
            if remaining < candidates[start_idx]:
                return False
                
            # If remaining is in the set of candidates from start_idx to k_idx_limit, we can pick it and finish.
            # This is a fast path.
            # But we need to ensure it's distinct and we haven't used it. Since we are picking indices in increasing order, and we are checking if `remaining` is in the slice, we just need to check if it exists in the slice.
            # However, `remaining` might be formed by multiple factors.
            
            # Let's check if `remaining` is directly in the available candidates.
            # If so, we can pick it.
            # But we must ensure we don't pick it if we already did? No, indices are unique.
            # If we pick `candidates[i]`, we move to `i+1`.
            # So if `remaining == candidates[i]`, we can return True (by picking it).
            
            for i in range(start_idx, k_idx_limit + 1):
                f = candidates[i]
                if f > remaining:
                    break # Since sorted, no larger f can divide remaining (unless remaining is 0, which it isn't)
                    # Actually, if f > remaining, remaining % f != 0.
                
                if remaining % f == 0:
                    # We can pick f
                    # If remaining / f == 1, we are done
                    # Else, we need to form remaining / f from candidates[i+1..k_idx_limit]
                    if exists_subset(remaining // f, i + 1, k_idx_limit):
                        return True
                    
            return False
            
        # Check if a solution exists with max factor K
        if exists_subset(target, 0, k_idx):
            # We found a solution. Now we need to retrieve it.
            # We can modify the DFS to return the path.
            # Or just reconstruct it.
            
            def find_subset(remaining, start_idx, k_idx_limit):
                if remaining == 1:
                    return []
                
                if start_idx > k_idx_limit:
                    return None
                
                for i in range(start_idx, k_idx_limit + 1):
                    f = candidates[i]
                    if f > remaining:
                        break
                    
                    if remaining % f == 0:
                        res = find_subset(remaining // f, i + 1, k_idx_limit)
                        if res is not None:
                            return [f] + res
                return None
            
            sol = find_subset(target, 0, k_idx)
            if sol:
                return sol
                
    return best_factors
# EVOLVE-BLOCK-END
