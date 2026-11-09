import numpy as np
from math import isqrt

def construct():
    N = 10000
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    P = len(primes)
    
    max_gap = N
    G = max_gap + 1
    
    # dp[i, g]: max length ending at primes[i] with last gap exactly g
    dp = np.zeros((P, G), dtype=np.int32)
    # prefix[i, g]: max length ending at primes[i] with last gap <= g
    prefix = np.zeros((P, G), dtype=np.int32)
    # argmax_prefix[i, g]: the gap g' <= g that achieves prefix[i, g]
    argmax_prefix = np.zeros((P, G), dtype=np.int32)
    # prev_idx[i, g]: the index of the previous prime in the optimal sequence ending at i with gap g
    prev_idx = np.full((P, G), -1, dtype=np.int32)
    # prev_gap[i, g]: the gap of the previous prime in the optimal sequence ending at i with gap g
    prev_gap = np.zeros((P, G), dtype=np.int32)
    
    # Initialize
    for i in range(P):
        dp[i, 0] = 1
        prefix[i, 0] = 1
        argmax_prefix[i, 0] = 0
        
    # Precompute primes array for vectorization
    primes_arr = np.array(primes)
    
    for j in range(1, P):
        p_j = primes_arr[j]
        # Gaps from all previous primes to p_j
        gaps = p_j - primes_arr[:j]
        
        # Get best lengths and gaps from previous rows
        # prefix[i, gaps[i]] is the max length ending at i with gap <= gaps[i]
        best_lens = prefix[:j, gaps]
        best_gaps = argmax_prefix[:j, gaps]
        
        new_lens = 1 + best_lens
        
        # Current values in dp[j, gaps]
        current_lens = dp[j, gaps]
        
        # Determine which updates are improvements
        mask = new_lens > current_lens
        
        # Update dp
        dp[j, gaps] = np.where(mask, new_lens, current_lens)
        
        # Update prev_idx and prev_gap
        # Indices of previous primes are 0 to j-1
        prev_indices = np.arange(j)
        prev_idx[j, gaps] = np.where(mask, prev_indices, prev_idx[j, gaps])
        prev_gap[j, gaps] = np.where(mask, best_gaps, prev_gap[j, gaps])
        
        # Update prefix and argmax_prefix for row j
        dp_row = dp[j, :]
        prefix_row = np.maximum.accumulate(dp_row)
        prefix[j, :] = prefix_row
        
        # Compute argmax_prefix for row j
        # prev_prefix values: [0, prefix_row[0], ..., prefix_row[G-2]]
        prev_prefix_vals = np.concatenate(([0], prefix_row[:-1]))
        is_new_max = dp_row >= prev_prefix_vals
        indices = np.arange(G)
        candidates = np.where(is_new_max, indices, -1)
        argmax_row = np.maximum.accumulate(candidates)
        argmax_prefix[j, :] = argmax_row

    # Find best ending
    max_len = np.max(prefix)
    best_idx, best_gap = np.unravel_index(np.argmax(prefix), prefix.shape)
    
    # Reconstruct
    seq = []
    idx = best_idx
    g = best_gap
    while g > 0:
        seq.append(primes[idx])
        idx = prev_idx[idx, g]
        g = prev_gap[idx, g]
    seq.append(primes[idx])
    seq.reverse()
    return seq
