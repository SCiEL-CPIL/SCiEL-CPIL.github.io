def construct():
    import time
    from math import isqrt
    from bisect import bisect_left

    deadline = time.time() + 80  # safety margin before 90s kill

    # Sieve of Eratosthenes for primes <= 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))

    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)

    if n < 2:
        return primes

    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j] (i < j)
    # The gap is primes[j] - primes[i].
    # Transition: dp[i][j] = 1 + max{ dp[k][i] for k < i such that primes[i] - primes[k] <= primes[j] - primes[i] }
    #             = 1 + max{ dp[k][i] for k < i such that primes[k] >= 2*primes[i] - primes[j] }
    
    # dp is a list of lists. dp[i] is a list of length n, but we only fill dp[i][j] for j > i.
    # Actually, we only need dp[k][i] for k < i.
    # Let's use a 2D array dp where dp[i][j] stores the max length ending at i, j.
    # Since n is small (~1229), n^2 is ~1.5M, which fits in memory.
    
    # dp[i][j] for i < j
    dp = [[0] * n for _ in range(n)]
    
    best_len = 1
    best_i = -1
    best_j = -1

    # Initialize base cases: any pair (i, j) with i < j has length at least 2.
    # We will compute dp[i][j] for all i < j.
    
    # To optimize, for each i, we can precompute a suffix maximum array of dp[k][i] for k < i.
    # Let suff[i][k] = max{ dp[m][i] for m in [k, i-1] }.
    # Then dp[i][j] = 1 + suff[i][k_min], where k_min is the smallest k such that primes[k] >= 2*primes[i] - primes[j].
    # If no such k exists in [0, i-1], then dp[i][j] = 2.

    # We process i from 0 to n-1.
    # For a fixed i, we need dp[k][i] for all k < i. These are already computed because k < i.
    # So we can build the suffix max array for column i (as the second index) before processing row i (as the first index).
    
    # Wait, the indices are (prev, curr).
    # dp[k][i] means sequence ends at k, i.
    # We want to extend to j. So we look at sequences ending at i.
    # The previous element was k. So we need dp[k][i].
    # Yes, for fixed i (current end), we look at all k < i (previous end).
    # So we need to query dp[k][i] for k < i.
    
    # Let's restructure:
    # Iterate i from 0 to n-1 (this is the second-to-last element in the pair (i, j)? No, i is the previous, j is current? No.)
    # Let's stick to: dp[i][j] where i < j.
    # To compute dp[i][j], we need max dp[k][i] for k < i with primes[k] >= 2*primes[i] - primes[j].
    # So for a fixed i, we need access to dp[k][i] for all k < i.
    # We can precompute, for each i, the suffix max of dp[k][i] over k.
    # Let suff_i[k] = max{ dp[m][i] for m in [k, i-1] }.
    # Then for a given j > i, we find k_min = lower_bound(2*primes[i] - primes[j]) in primes[0:i].
    # If k_min < i, dp[i][j] = 1 + suff_i[k_min].
    # Else dp[i][j] = 2.
    
    # We need to compute suff_i for each i.
    # suff_i depends on dp[k][i] for k < i.
    # So we can compute suff_i after we have computed all dp[k][i] for k < i.
    # This means we should iterate i in increasing order.
    # For each i:
    #   1. Build suff_i from dp[k][i] for k < i.
    #   2. For each j > i, compute dp[i][j] using suff_i.
    
    # Initialize dp[k][i] for k < i.
    # For i=0, no k < i.
    # For i=1, k=0. dp[0][1] = 2 (base case, no previous gap constraint).
    # Actually, for any pair (i, j), the length is at least 2.
    # So we should initialize dp[i][j] = 2 for all i < j?
    # Or handle it in the logic: if no valid k, then length is 2.
    
    # Let's trace:
    # i=0: suff_0 is empty. For j > 0, k_min will be >= 0 (since primes[0] is the smallest).
    # Wait, if i=0, there are no k < 0. So the max is over an empty set.
    # So dp[0][j] should be 2.
    # My logic: if k_min < i, then ... else 2.
    # For i=0, i is 0. k_min < 0 is false. So dp[0][j] = 2. Correct.
    
    # i=1: suff_1[0] = dp[0][1].
    # We need dp[0][1] to be computed.
    # In the loop for i=0, we computed dp[0][1].
    # So yes, this order works.

    # Precompute suffix max arrays? Or compute on the fly?
    # Computing on the fly for each i is O(i). Total O(n^2).
    # Querying is O(log i) for binary search. Total O(n^2 log n).
    # n ~ 1229. n^2 ~ 1.5M. log n ~ 11. 1.5M * 11 ~ 16.5M operations.
    # Python might be slow, but 16M simple ops should take < 1s.
    # Let's try to optimize the inner loop.

    # dp[i][j] = 2 if no valid k.
    # We store dp in a 2D list.
    
    # We also need to track the predecessor for reconstruction.
    # pred[i][j] = the k that gave the max dp[k][i].
    # If multiple k give the same max, pick any (or the one with largest k? doesn't matter for length).
    
    pred = [[-1] * n for _ in range(n)]

    for i in range(n):
        if time.time() > deadline:
            break
            
        # Build suffix max for column i (as second index)
        # suff[k] = max{ dp[m][i] for m in [k, i-1] }
        # We also want to track which m gave the max.
        # suff_val[k], suff_idx[k]
        
        if i == 0:
            # No k < 0
            pass
        else:
            # suff_val is an array of length i
            # suff_val[k] = max dp[m][i] for m in [k, i-1]
            suff_val = [0] * i
            suff_idx = [-1] * i
            
            # Initialize for k = i-1
            suff_val[i-1] = dp[i-1][i]
            suff_idx[i-1] = i-1
            
            for k in range(i-2, -1, -1):
                if dp[k][i] > suff_val[k+1]:
                    suff_val[k] = dp[k][i]
                    suff_idx[k] = k
                else:
                    suff_val[k] = suff_val[k+1]
                    suff_idx[k] = suff_idx[k+1]

        pi = primes[i]
        
        for j in range(i + 1, n):
            T = 2 * pi - primes[j]
            
            # Find smallest k in [0, i-1] such that primes[k] >= T
            # If i == 0, no k.
            if i == 0:
                dp[i][j] = 2
                pred[i][j] = -1
            else:
                # Binary search in primes[0:i] for T
                # bisect_left(primes, T, 0, i)
                k_min = bisect_left(primes, T, 0, i)
                
                if k_min < i:
                    # Valid k exists in [k_min, i-1]
                    best_k = suff_idx[k_min]
                    dp[i][j] = 1 + suff_val[k_min]
                    pred[i][j] = best_k
                else:
                    dp[i][j] = 2
                    pred[i][j] = -1
                    
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j

    # Reconstruct
    if best_len <= 1:
        return [primes[0]]
        
    seq = [primes[best_j], primes[best_i]]
    i = best_i
    j = best_j
    
    while pred[i][j] != -1:
        k = pred[i][j]
        seq.append(primes[k])
        j = i
        i = k
        
    seq.reverse()
    return seq
