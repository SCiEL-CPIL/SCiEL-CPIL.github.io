"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

def construct():
    """
    Find 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    target = math.factorial(12)
    # 12! = 479001600
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7^1
    
    # We want to find distinct integers > 12 whose product is target.
    # We want to minimize the maximum factor.
    
    # Strategy: Use backtracking/DFS over the prime exponent vectors.
    # We'll generate divisors of target that are > 12, and try to partition
    # the prime exponents into a set of distinct divisors > 12.
    
    # The total prime exponents are:
    # 2: 10, 3: 5, 5: 2, 7: 1
    
    # We need to assign these exponents to factors. Each factor is a divisor of target.
    # All factors must be > 12 and distinct.
    # We want to minimize the largest factor.
    
    # Approach: 
    # 1. Generate all divisors of target that are > 12.
    # 2. Sort them.
    # 3. Use a search that tries to find a set of distinct divisors > 12 
    #    whose product is target, minimizing the max.
    
    # Since the number of divisors is manageable, we can enumerate them.
    # 12! = 2^10 * 3^5 * 5^2 * 7^1
    # Number of divisors = (10+1)*(5+1)*(2+1)*(1+1) = 11*6*3*2 = 396
    
    # Let's generate all divisors.
    divisors = []
    for a in range(11):  # exponent of 2: 0..10
        for b in range(6):  # exponent of 3: 0..5
            for c in range(3):  # exponent of 5: 0..2
                for d in range(2):  # exponent of 7: 0..1
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if val > 12:
                        divisors.append((val, a, b, c, d))
    
    # Sort by value
    divisors.sort()
    
    # We want to find a subset of these divisors (with distinct values) 
    # such that their product is target, and the exponents sum to (10,5,2,1).
    # We want to minimize the maximum value in the subset.
    
    # Since we want to minimize the max, we can try max_value in ascending order.
    # For each candidate max M, we check if there's a valid decomposition using
    # only divisors <= M.
    
    # But we also need at most 12 factors.
    
    # Let's use a recursive search with memoization on the remaining exponents.
    # State: (remaining_2, remaining_3, remaining_5, remaining_7, min_index, max_allowed)
    # where min_index is the index in divisors to start from (to ensure distinctness and ordering)
    
    # Actually, since we need distinct factors, and we're iterating through sorted divisors,
    # we can ensure distinctness by only picking each divisor at most once.
    
    # Let's try a different approach: 
    # For a given upper bound M on the largest factor, check if we can decompose target
    # into distinct divisors all <= M and > 12.
    
    # We'll do this with backtracking. The state is the remaining exponents.
    
    def can_decompose(rem_2, rem_3, rem_5, rem_7, start_idx, max_val, factors_used):
        """
        Try to decompose remaining exponents into distinct divisors from divisors[start_idx:]
        that are <= max_val.
        """
        if time.time() > deadline:
            return None
        
        if rem_2 == 0 and rem_3 == 0 and rem_5 == 0 and rem_7 == 0:
            return []
        
        if factors_used >= 12:
            return None
        
        # If remaining product is 1, we're done
        # But we need to check if we can reach exactly the target
        
        # Try each divisor from start_idx onwards
        for i in range(start_idx, len(divisors)):
            val, a, b, c, d = divisors[i]
            if val > max_val:
                break
            
            # Check if this divisor can be used
            if a <= rem_2 and b <= rem_3 and c <= rem_5 and d <= rem_7:
                # Use this divisor
                result = can_decompose(rem_2 - a, rem_3 - b, rem_5 - c, rem_7 - d, i + 1, max_val, factors_used + 1)
                if result is not None:
                    return [val] + result
        
        return None
    
    # Try different max values in ascending order
    # The minimum possible max is at least 13 (smallest divisor > 12)
    # The maximum possible max is target itself (single factor)
    
    # Let's try each unique divisor value as a candidate for the maximum
    unique_vals = sorted(set(v for v, _, _, _, _ in divisors))
    
    # Also, the single factor case: max = target
    # But we want to minimize, so start small
    
    best_result = None
    best_max = target + 1
    
    for max_val in unique_vals:
        if time.time() > deadline:
            break
        
        # Check if we can decompose with all factors <= max_val
        result = can_decompose(10, 5, 2, 1, 0, max_val, 0)
        if result is not None:
            # Found a valid decomposition
            if max_val < best_max:
                best_max = max_val
                best_result = result
            
            # Since we're going in ascending order, the first valid one is optimal
            # But wait, we need to make sure we found the minimum max
            # Since we're iterating max_val in ascending order, the first valid decomposition
            # has the minimum possible max
            break
    
    if best_result is None:
        # Fallback: single factor
        return [target]
    
    return best_result
# EVOLVE-BLOCK-END
