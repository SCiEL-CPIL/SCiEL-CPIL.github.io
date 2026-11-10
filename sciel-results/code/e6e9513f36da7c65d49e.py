"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def compute_dp(S):
    """
    Compute the set of achievable sums for each cardinality using bitsets.
    Returns a list of integers where the i-th integer is a bitset of sums achievable by subsets of size i.
    """
    m = len(S)
    if m == 0:
        return [1]  # dp[0] = bit 0 set
    
    dp = [0] * (m + 1)
    dp[0] = 1  # bit 0 is set for cardinality 0
    
    for x in S:
        # Iterate backwards to ensure each element is used at most once
        for k in range(m, 0, -1):
            # Shift the bitset for cardinality k-1 by x and OR into cardinality k
            dp[k] |= (dp[k-1] << x)
            
    return dp

def count_collisions(dp):
    """
    Count the total number of sum collisions across different cardinalities.
    A collision is a sum value that is achievable by subsets of two different cardinalities.
    """
    m = len(dp) - 1
    collisions = 0
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            common = dp[i] & dp[j]
            if common:
                collisions += common.bit_count()
    return collisions

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    
    Strategy:
    1. Start with structured random sets (not purely random).
    2. Use global conflict counting to identify most problematic elements.
    3. Remove elements involved in most cross-cardinality collisions.
    4. Multiple restarts with different initialization strategies.
    """
    max_size = 60
    max_val = 256
    best_set = []
    
    # Multiple initialization strategies
    for restart in range(30):
        if restart < 10:
            # Strategy 1: Random sampling
            S = random.sample(range(1, max_val + 1), max_size)
        elif restart < 20:
            # Strategy 2: Larger values tend to be better (less overlap)
            # Sample from upper half with some randomness
            upper = list(range(129, max_val + 1))
            lower = list(range(1, 129))
            S = random.sample(upper, 40) + random.sample(lower, 20)
            random.shuffle(S)
        else:
            # Strategy 3: Geometric-like spacing
            # Start with a base and add increasing gaps
            base = random.randint(1, 50)
            S = []
            current = base
            while len(S) < max_size and current <= max_val:
                S.append(current)
                gap = random.randint(max(1, current // 3), max(2, current // 2))
                current += gap
            # Fill remaining with random values not in S
            if len(S) < max_size:
                available = [x for x in range(1, max_val + 1) if x not in set(S)]
                S.extend(random.sample(available, max_size - len(S)))
            S = S[:max_size]
        
        current_size = len(S)
        S = sorted(S)
        
        # Global conflict-driven removal loop
        while current_size > 0:
            dp = compute_dp(S)
            colls = count_collisions(dp)
            
            if colls == 0:
                if len(S) > len(best_set):
                    best_set = S.copy()
                break
            
            # Compute conflict contribution of each element
            # An element x contributes to collisions if removing it reduces collision count
            # We estimate by checking which cardinality pairs have collisions involving sums that include x
            
            # More efficient: try removing each element and measure collision reduction
            # But this is O(n^2) per step, so we do it smartly
            
            # First, try to find elements that are in many collision sums
            # For each collision sum s (achievable at cardinality i and j, i!=j),
            # identify which elements could be part of subsets producing s
            
            # Simpler approach: greedy removal with collision delta
            min_colls_after_removal = colls + 1
            best_idx_to_remove = None
            
            # Sample a subset of elements to evaluate if set is large
            if current_size > 20:
                # Evaluate only top candidates based on value (larger values often cause more issues)
                indices_to_try = list(range(current_size))
                random.shuffle(indices_to_try)
                indices_to_try = indices_to_try[:min(15, current_size)]
            else:
                indices_to_try = list(range(current_size))
            
            for idx in indices_to_try:
                S_temp = S[:idx] + S[idx+1:]
                dp_temp = compute_dp(S_temp)
                colls_temp = count_collisions(dp_temp)
                
                if colls_temp < min_colls_after_removal:
                    min_colls_after_removal = colls_temp
                    best_idx_to_remove = idx
                    
                    if colls_temp == 0:
                        break
            
            if best_idx_to_remove is None:
                # No improvement found in sampled indices, try full search if small
                if current_size <= 15:
                    for idx in range(current_size):
                        S_temp = S[:idx] + S[idx+1:]
                        dp_temp = compute_dp(S_temp)
                        colls_temp = count_collisions(dp_temp)
                        if colls_temp < min_colls_after_removal:
                            min_colls_after_removal = colls_temp
                            best_idx_to_remove = idx
                            if colls_temp == 0:
                                break
            
            if best_idx_to_remove is None:
                break
            
            S.pop(best_idx_to_remove)
            current_size -= 1
            
            if min_colls_after_removal == 0:
                if len(S) > len(best_set):
                    best_set = S.copy()
                break
    
    return list(best_set)

# EVOLVE-BLOCK-END
