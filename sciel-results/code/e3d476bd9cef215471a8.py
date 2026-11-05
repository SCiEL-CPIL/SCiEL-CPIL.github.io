# EVOLVE-BLOCK-START
import numpy as np

def _check_subset_sum(target, diffs, min_size):
    """
    Returns True if there exists a subset of 'diffs' with size >= min_size 
    that sums to 'target'.
    Uses bitset DP for speed.
    """
    n_diffs = len(diffs)
    if min_size > n_diffs:
        return False
    if min_size == 0:
        return target == 0 # Only empty set sums to 0 with size 0? 
                           # But usually min_size >= 1 in our usage.
                           # If min_size == 0, empty set is valid.
                           # However, problem states "subset of at least two OTHER elements".
                           # So min_size will be 1 or 2.
    
    # Quick check: if target is 0 and min_size == 0, True.
    # But we always pass min_size >= 1.
    
    OFFSET = 32768
    # Check if target is within reasonable bounds
    # Max sum magnitude: 48 * 512 = 24576.
    if target < -25000 or target > 25000:
        return False
        
    target_idx = target + OFFSET
    if target_idx < 0 or target_idx >= (1 << 16):
        return False

    # dp[k] is a bitset of sums achievable with exactly k elements.
    # We only need to track up to min_size? No, we need to know if ANY k >= min_size works.
    # But we can just compute all k up to n_diffs.
    
    # Optimization: If we just want to know if ANY subset of size >= min_size sums to target,
    # we can't easily stop early in the DP unless we check at the end.
    
    # Initialize
    # dp[0] has bit OFFSET set.
    dp = [0] * (n_diffs + 1)
    dp[0] = 1 << OFFSET
    
    for d in diffs:
        # Iterate backwards
        # We only need to update up to the current max possible count
        # Let's just iterate all k backwards
        for k in range(n_diffs - 1, -1, -1):
            if dp[k]:
                if d > 0:
                    dp[k+1] |= (dp[k] << d)
                elif d < 0:
                    dp[k+1] |= (dp[k] >> (-d))
                else:
                    dp[k+1] |= dp[k]
            
    # Check result
    for k in range(min_size, n_diffs + 1):
        if dp[k] & (1 << target_idx):
            return True
            
    return False

def construct():
    n = 512
    max_card = 48
    S = []
    
    for x in range(1, n + 1):
        valid = True
        
        # Check 1: Is x the mean of a subset of S with size >= 2?
        # Equivalent to: 0 is a subset sum of [s - x for s in S] with size >= 2.
        if len(S) >= 2:
            diffs = [s - x for s in S]
            if _check_subset_sum(0, diffs, 2):
                valid = False
        
        if valid:
            # Check 2: For each a in S, is a the mean of a subset of S U {x} \ {a} with size >= 2?
            # New subsets must include x.
            # T = T0 U {x}, T0 subset S \ {a}, |T0| >= 1.
            # Sum(T0) + x = (|T0| + 1) * a
            # Sum(t - a for t in T0) = a - x
            # Let diffs_a = [s - a for s in S if s != a]
            # Check if (a - x) is a subset sum of diffs_a with size >= 1.
            for a in S:
                diffs_a = [s - a for s in S if s != a]
                target = a - x
                if _check_subset_sum(target, diffs_a, 1):
                    valid = False
                    break
        
        if valid:
            S.append(x)
            if len(S) >= max_card:
                break
                
    return S
# EVOLVE-BLOCK-END
