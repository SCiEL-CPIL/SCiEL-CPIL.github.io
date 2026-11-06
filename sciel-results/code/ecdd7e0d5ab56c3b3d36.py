"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import math
import time

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin from 60s limit (evaluator kills at 90s, but we want to be safe)
    
    target = math.factorial(12)
    
    # Generate all divisors of target
    all_divs = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            all_divs.add(i)
            all_divs.add(target // i)
    
    # Filter divisors > 12
    candidates = sorted([d for d in all_divs if d > 12])
    n_candidates = len(candidates)
    
    # Precompute a set for O(1) membership check
    candidate_set = set(candidates)
    
    best_factors = [target]
    best_max = target
    
    # We want to find a subset of distinct candidates from `candidates` that multiply to target.
    # To minimize the largest factor, we search for solutions with smaller max factors first?
    # Or just find any valid solution and track the min max.
    # Since the number of factors is variable (1 to 12), and distinct.
    
    # DFS approach with strong pruning.
    # State: remaining product, index in candidates to start from (to ensure distinctness), 
    #        current list of factors, current max.
    
    def dfs(remaining, start_idx, current_factors, current_max):
        nonlocal best_factors, best_max
        
        if time.time() - start_time > timeout:
            return
            
        if remaining == 1:
            # Found a solution
            if current_max < best_max:
                best_max = current_max
                best_factors = current_factors[:]
            return
            
        # If no more factors can be picked
        if start_idx >= n_candidates:
            return
            
        # Pruning 1: If remaining is in candidates[start_idx:], we can pick it as the last factor.
        # But we must ensure it's distinct (which is handled by start_idx) and > 12 (handled by candidates).
        # If remaining is a candidate >= start_idx, and remaining < best_max, we have a valid solution.
        # Actually, if remaining is a candidate, we can just add it and finish.
        if remaining in candidate_set:
            # Check if remaining is available (index >= start_idx)
            # Since candidates are sorted, if remaining > candidates[start_idx-1] (or start_idx==0), it might be available.
            # Better: check if remaining is in candidates[start_idx:]
            # O(n) check is fine for small n, but let's use binary search or just check.
            # Since we are iterating, we can just check if remaining is a candidate.
            # The index check is implicit if we just append and return?
            # No, we need to ensure we don't reuse a factor already used.
            # Since we are picking in increasing order of index, and `remaining` is the product of the rest,
            # if `remaining` is a candidate, we can pick it IF its index is >= start_idx.
            
            # Find index of remaining in candidates
            # Since candidates is sorted, we can use bisect.
            import bisect
            idx = bisect.bisect_left(candidates, remaining)
            if idx < n_candidates and candidates[idx] == remaining and idx >= start_idx:
                # We can pick remaining as the last factor
                new_max = max(current_max, remaining)
                if new_max < best_max:
                    # This is a valid solution
                    best_max = new_max
                    best_factors = current_factors + [remaining]
                return
        
        # Pruning 2: If remaining is prime and > 12, it must be a single factor.
        # If it's not in candidates[start_idx:], we can't form it.
        # But checking primality is expensive. 
        # However, if remaining < 182 (13*14), it can't be formed by 2+ factors.
        # So if remaining < 182 and remaining != 1, it must be a single factor.
        # We already handled the case where remaining is a candidate above.
        # If remaining < 182 and remaining is NOT a candidate >= start_idx, we can prune.
        if remaining < 182:
            return
        
        # Pruning 3: The smallest possible product of (12 - len(current_factors)) distinct factors > 12
        # starting from start_idx.
        # This is complex. A simpler bound:
        # The smallest factor we can pick is candidates[start_idx].
        # If we pick k more factors, the product is at least candidates[start_idx] * (candidates[start_idx]+1) * ...
        # But they don't have to be consecutive.
        # A looser bound: If remaining < candidates[start_idx], we can't even pick the smallest available factor.
        # But we are picking factors that divide remaining.
        
        # Let's iterate through candidates from start_idx to n_candidates - 1
        for i in range(start_idx, n_candidates):
            f = candidates[i]
            
            # If f is greater than or equal to best_max, we can break?
            # No, because candidates are sorted ascending. If f >= best_max, any subsequent f' will also be >= best_max.
            # And since f is part of the product, the max will be at least f.
            # So if f >= best_max, we can break the loop.
            if f >= best_max:
                break
                
            if remaining % f == 0:
                new_remaining = remaining // f
                new_max = max(current_max, f)
                
                # If new_max >= best_max, we can skip this branch entirely?
                # Yes, because any completion will have max >= new_max >= best_max.
                if new_max < best_max:
                    current_factors.append(f)
                    dfs(new_remaining, i + 1, current_factors, new_max)
                    current_factors.pop()
                    
                # Additional pruning: if new_remaining > 1 and new_remaining < candidates[i+1] (if exists)
                # and new_remaining is not a candidate >= i+1, then we can't complete?
                # If new_remaining < candidates[i+1], it cannot be formed by a single factor from i+1 onwards.
                # Can it be formed by multiple? 
                # The smallest product of two factors > 12 is 13*14 = 182.
                # So if new_remaining < 182, it cannot be formed by 2 or more factors.
                # Thus, if new_remaining < 182 and new_remaining != 1, it must be a single factor.
                # So if new_remaining < 182 and new_remaining is not in candidates[i+1:], we can prune.
                
                if new_remaining > 1:
                    if new_remaining < 182:
                        # Must be a single factor
                        # Check if new_remaining is a candidate >= i+1
                        import bisect
                        idx_rem = bisect.bisect_left(candidates, new_remaining)
                        if idx_rem >= n_candidates or candidates[idx_rem] != new_remaining or idx_rem < i + 1:
                            continue
                    else:
                        # Could be single or multiple.
                        # If new_remaining is not in candidates[i+1:], we need at least 2 factors.
                        # The smallest product of 2 distinct factors > 12 is 13*14 = 182.
                        # If new_remaining < 182, it can't be 2 factors.
                        # If new_remaining >= 182, it might be 2 factors.
                        pass
                        
    dfs(target, 0, [], 0)
    
    return best_factors

# EVOLVE-BLOCK-END
