import numpy as np
from math import isqrt

def construct():
    limit = 10000
    # Sieve of Eratosthenes
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array(np.where(is_prime)[0])
    N = len(primes)
    
    if N == 0:
        return []
        
    max_gap = int(primes[-1] - primes[0])
    
    # dp[i, g] = length of longest sequence ending at primes[i] with last gap g
    # pref[i, g] = max(dp[i, 0..g])
    # best_gap[i, g] = argmax(dp[i, 0..g])
    
    dp = np.zeros((N, max_gap + 1), dtype=np.int32)
    pref = np.zeros((N, max_gap + 1), dtype=np.int32)
    best_gap = np.zeros((N, max_gap + 1), dtype=np.int32)
    
    # Map prime value to index for reconstruction
    prime_to_idx = {int(p): i for i, p in enumerate(primes)}
    
    for i in range(N):
        p_i = primes[i]
        # Base case: sequence of length 1, gap 0
        dp[i, 0] = 1
        pref[i, 0] = 1
        best_gap[i, 0] = 0
        
        if i > 0:
            # Calculate gaps to all previous primes
            prev_primes = primes[:i]
            gaps = p_i - prev_primes
            valid = gaps > 0
            g_v = gaps[valid]
            j_v = np.arange(i)[valid]
            
            # Extend sequences from previous primes
            # We need max length ending at j_v with gap <= g_v
            vals = pref[j_v, g_v]
            dp[i, g_v] = vals + 1
            
        # Compute prefix maximums and best gaps for this prime
        if max_gap > 0:
            cum_max = np.maximum.accumulate(dp[i])
            pref[i] = cum_max
            
            # Find the index of the maximum value in the prefix
            # We do this in a loop for simplicity and correctness
            bg = np.zeros(max_gap + 1, dtype=np.int32)
            curr_best_val = int(dp[i, 0])
            curr_best_idx = 0
            bg[0] = 0
            
            # Optimization: only iterate up to the max gap relevant for this prime?
            # No, max_gap is global. But dp[i, g] is 0 for large g if no prev prime.
            # However, pref[i, g] stays constant.
            # We can break early if dp[i, g] is 0 and curr_best_val is stable?
            # But we need bg for all g up to max_gap because future primes might query any g.
            # So we must fill the whole array.
            
            for g in range(1, max_gap + 1):
                val = int(dp[i, g])
                if val > curr_best_val:
                    curr_best_val = val
                    curr_best_idx = g
                bg[g] = curr_best_idx
            best_gap[i] = bg
            
    # Find the global maximum length
    max_val = int(np.max(dp))
    # Get the first occurrence of the maximum
    flat_idx = np.argmax(dp)
    i_star, g_star = np.unravel_index(flat_idx, dp.shape)
    
    # Reconstruct the sequence
    seq = [int(primes[i_star])]
    curr_i = i_star
    curr_g = int(g_star)
    
    while curr_g > 0:
        p_prev = int(primes[curr_i]) - curr_g
        j = prime_to_idx[p_prev]
        seq.append(p_prev)
        curr_g = int(best_gap[j][curr_g])
        curr_i = j
        
    seq.reverse()
    return seq
