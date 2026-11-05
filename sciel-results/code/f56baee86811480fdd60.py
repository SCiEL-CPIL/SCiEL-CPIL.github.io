"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    max_gap = limit
    
    # dp[i][g]: max length of sequence ending at primes[i] with last gap g
    # prev_idx[i][g]: index j of the previous prime
    # prev_gap[i][g]: gap h of the sequence ending at primes[j]
    
    # Using lists of lists for simplicity and speed in Python for this size
    # n ~ 1229, max_gap ~ 10000
    # Memory: 1229 * 10000 * 3 * 8 bytes ~ 300MB? 
    # Python ints are larger. 1229 * 10000 * 3 objects.
    # This might be too slow/memory heavy in pure Python lists.
    # Let's use arrays from array module or numpy if available.
    # The prompt says NumPy is available.
    
    import numpy as np
    
    # dp[i] is an array of int16 or int32. Max length is small (< 100). int16 is enough.
    # prev_idx needs int16 (n < 32767).
    # prev_gap needs int16 (gap < 10000).
    
    dp = np.zeros((n, max_gap + 1), dtype=np.int16)
    prev_idx = np.full((n, max_gap + 1), -1, dtype=np.int16)
    prev_gap = np.full((n, max_gap + 1), -1, dtype=np.int16)
    
    # pref_max_val[i][g] = max(dp[i][0..g])
    # pref_max_idx[i][g] = argmax index h
    pref_max_val = np.zeros((n, max_gap + 1), dtype=np.int16)
    pref_max_idx = np.full((n, max_gap + 1), -1, dtype=np.int16)
    
    # Initialize base case: sequence of length 1
    # We treat gap 0 as the "no gap" state with length 1.
    # This ensures that any extension has a valid predecessor of length 1.
    dp[:, 0] = 1
    
    # Precompute prefix max for i=0?
    # Actually, we compute prefix max for i after processing all j < i.
    # But for i=0, there are no j < 0.
    # So we just compute prefix max for i=0 from the initialized dp[0].
    
    for i in range(n):
        if time.time() > deadline:
            break
            
        # Process transitions from all j < i to i
        for j in range(i):
            g = primes[i] - primes[j]
            # Get best length and gap from j with gap <= g
            best_len = pref_max_val[j, g]
            best_gap = pref_max_idx[j, g]
            
            cand_len = best_len + 1
            if cand_len > dp[i, g]:
                dp[i, g] = cand_len
                prev_idx[i, g] = j
                prev_gap[i, g] = best_gap
        
        # Compute prefix max for i
        # We can do this in a loop or using numpy cumulative max
        # But we need the index of the max as well.
        # Numpy doesn't have cummax with index directly.
        # Loop is O(G) = 10000. n * G = 12 million. Fast enough.
        
        current_max_val = 0
        current_max_idx = -1
        for g in range(max_gap + 1):
            if dp[i, g] > current_max_val:
                current_max_val = dp[i, g]
                current_max_idx = g
            pref_max_val[i, g] = current_max_val
            pref_max_idx[i, g] = current_max_idx

    # Find the best sequence
    # Max over all dp[i][g]
    max_len = 0
    best_i = 0
    best_g = 0
    
    # We can just look at the last row? No, the best sequence might end at any prime.
    # However, since we want to maximize length, and gaps are positive,
    # a longer sequence is generally "better".
    # We just need the global max.
    
    # Flatten and find max
    # To avoid huge memory copy, iterate
    for i in range(n):
        row_max = np.max(dp[i])
        if row_max > max_len:
            max_len = row_max
            best_i = i
            # Find the gap that achieved this max
            # There might be multiple. Any is fine.
            # We need the one that is part of a valid chain.
            # Since we updated dp, any g with dp[i][g]==max_len is valid.
            # Let's find the first one.
            best_g = np.argmax(dp[i])
            # Note: argmax returns the first occurrence.
            # If dp[i][best_g] is not max_len (due to tie-breaking or logic), 
            # we should check. But np.max and np.argmax are consistent.
            
    # Reconstruct path
    path = [primes[best_i]]
    curr_i = best_i
    curr_g = best_g
    
    # Safety loop
    while curr_g > 0:
        if time.time() > deadline:
            break
        j = prev_idx[curr_i, curr_g]
        if j < 0:
            break
        path.append(primes[j])
        curr_i = j
        curr_g = prev_gap[curr_i, curr_g]
        
        if curr_g < 0:
            break

    path.reverse()
    return path

# EVOLVE-BLOCK-END
