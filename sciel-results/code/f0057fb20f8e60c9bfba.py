"""Program finding long increasing prime sequences with nondecreasing gaps <= 10000."""
# EVOLVE-BLOCK-START
import time
import sys
from math import isqrt

def construct():
    """
    Find the longest increasing subsequence of primes <= 10000 such that the gaps
    between consecutive elements are non-decreasing.
    
    Returns a list of prime integers representing the longest such sequence.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Generate primes up to 10000
    n_max = 10000
    is_prime = bytearray(b'\x01') * (n_max + 1)
    is_prime[0] = 0
    if n_max >= 1:
        is_prime[1] = 0
    for i in range(2, isqrt(n_max) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(range(i*i, n_max+1, i)))
    
    primes = [i for i in range(2, n_max + 1) if is_prime[i]]
    m = len(primes)
    
    # 2. Dynamic Programming
    # dp[i][j] = max length of valid chain ending at (primes[i], primes[j]) with i < j
    # We store dp as a list of lists for flexibility, but optimize inner loops.
    # Actually, let's use numpy for speed if possible, but lists are safer for small m.
    # m is ~1229. m^2 is ~1.5M. This fits in memory easily.
    
    # To maximize speed, we will iterate j from 1 to m-1.
    # For each j, we iterate i from 0 to j-1.
    # Condition: primes[j] - primes[i] >= primes[i] - primes[k] for some k < i.
    # This is equivalent to primes[k] >= 2*primes[i] - primes[j].
    
    # Let's try a different DP formulation that might be faster or simpler to implement efficiently.
    # Standard LIS style:
    # Let L[i] be the longest valid sequence ending at prime i.
    # This doesn't work directly because the constraint depends on the *previous* gap, 
    # not just the previous element. We need to know the last two elements.
    
    # So dp[i][j] is the correct state.
    # dp[i][j] = 2 if no valid k < i exists such that primes[i] - primes[k] <= primes[j] - primes[i].
    # dp[i][j] = max(dp[k][i] + 1) for all k < i satisfying primes[i] - primes[k] <= primes[j] - primes[i].
    
    # Optimization:
    # For a fixed j, we want to compute dp[i][j] for all i < j.
    # The condition is primes[k] >= 2*primes[i] - primes[j].
    # Let target = 2*primes[i] - primes[j].
    # We need the max value of dp[k][i] for k < i where primes[k] >= target.
    
    # We can precompute for each i, a data structure to query max dp[k][i] for k with primes[k] >= T.
    # Since primes are sorted, if we process i in increasing order, we can maintain a suffix max array for dp[:, i].
    
    # Let's create an array M[i] of length i, where M[i][k] = max(dp[x][i] for x in [k, i-1]).
    # Then dp[i][j] = M[i][lo_index] + 1, where lo_index is the smallest index k < i such that primes[k] >= target.
    # If no such k exists, dp[i][j] = 2.
    
    # We will store dp as a 2D list or numpy array.
    # Let's use numpy arrays for dp and the suffix max structures.
    
    import numpy as np
    
    dp = np.zeros((m, m), dtype=np.int16)
    # best_k[i][j] stores the index k that gave the max value for dp[i][j]
    best_k = np.full((m, m), -1, dtype=np.int16)
    
    # suffix_max[i] is an array of length i, where suffix_max[i][k] = max(dp[x][i] for x in k..i-1)
    # suffix_arg[i] is an array of length i, where suffix_arg[i][k] = argmax x in k..i-1
    suffix_max = [None] * m
    suffix_arg = [None] * m
    
    max_len = 0
    max_i = -1
    max_j = -1
    
    for j in range(1, m):
        if time.time() > deadline:
            break
            
        p_j = primes[j]
        
        # Precompute lo_indices for all i < j
        # target_i = 2 * primes[i] - p_j
        # We need the smallest index k < i such that primes[k] >= target_i.
        
        # We can vectorize the target calculation
        targets = 2 * primes[:j] - p_j
        
        # For each i, find the lower bound in primes[0:i] for targets[i]
        # np.searchsorted works on the whole array, but we restrict to [0, i).
        # This is hard to vectorize directly with searchsorted because the upper bound changes.
        # However, we can just do a loop for i, and use bisect.
        
        from bisect import bisect_left
        
        for i in range(j):
            target = targets[i]
            
            # Find smallest k < i such that primes[k] >= target
            # bisect_left(primes, target) gives the global index.
            # We need to ensure the found index is < i.
            # If the global index is < i, then it's a valid candidate.
            # If the global index is >= i, then no k < i satisfies the condition.
            
            k_candidate = bisect_left(primes, target)
            
            if k_candidate < i:
                # We have at least one valid k in [k_candidate, i-1].
                # We need the max dp[x][i] for x in [k_candidate, i-1].
                # This is stored in suffix_max[i][k_candidate].
                # But we need to compute suffix_max[i] first?
                # No, suffix_max[i] depends on dp[x][i] for x < i.
                # dp[x][i] was computed in the previous iteration of j (when j was i).
                # So suffix_max[i] is already available.
                
                if suffix_max[i] is not None:
                    val = suffix_max[i][k_candidate]
                    if val > 0:
                        dp[i, j] = val + 1
                        best_k[i, j] = suffix_arg[i][k_candidate]
                    else:
                        # This case shouldn't happen if suffix_max is correct, 
                        # as dp[x][i] is at least 2 for valid x.
                        # Wait, dp[x][i] is 2 if no predecessor.
                        # So val should be at least 2 if there is any valid x.
                        # If k_candidate < i, there is at least one x.
                        # So val >= 2.
                        dp[i, j] = val + 1
                        best_k[i, j] = suffix_arg[i][k_candidate]
                else:
                    # i=0? No, i < j, and j>=1, so i can be 0.
                    # If i=0, suffix_max[0] is None.
                    # But k_candidate < 0 is impossible.
                    # So this branch is not reached for i=0.
                    dp[i, j] = 2
            else:
                dp[i, j] = 2
                best_k[i, j] = -1
            
            if dp[i, j] > max_len:
                max_len = dp[i, j]
                max_i = i
                max_j = j
        
        # Compute suffix_max[j] and suffix_arg[j]
        # suffix_max[j][k] = max(dp[x][j] for x in k..j-1)
        if j > 0:
            sm = np.zeros(j, dtype=np.int16)
            sa = np.zeros(j, dtype=np.int16)
            
            # Start from the right
            current_max = 0
            current_arg = -1
            
            for k in range(j-1, -1, -1):
                if dp[k, j] > current_max:
                    current_max = dp[k, j]
                    current_arg = k
                sm[k] = current_max
                sa[k] = current_arg
            
            suffix_max[j] = sm
            suffix_arg[j] = sa
    
    if max_len <= 2:
        # Fallback
        if len(primes) >= 2:
            return [primes[0], primes[1]]
        else:
            return primes
    
    # Reconstruct the sequence
    seq = [primes[max_j]]
    i = max_i
    j = max_j
    length = max_len
    
    while length > 1:
        k = best_k[i, j]
        if k < 0:
            break
        seq.append(primes[k])
        j = i
        i = k
        length -= 1
    
    seq.reverse()
    
    # Verify
    # Check non-decreasing gaps
    valid = True
    for idx in range(2, len(seq)):
        if seq[idx] - seq[idx-1] < seq[idx-1] - seq[idx-2]:
            valid = False
            break
    
    if valid:
        return seq
    else:
        # Fallback greedy if reconstruction failed (shouldn't happen)
        res = [primes[0]]
        if len(primes) > 1:
            res.append(primes[1])
            last_gap = primes[1] - primes[0]
            for p in primes[2:]:
                if p - res[-1] >= last_gap:
                    res.append(p)
                    last_gap = p - res[-1]
        return res

# EVOLVE-BLOCK-END
