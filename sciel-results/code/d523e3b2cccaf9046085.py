"""Rewritten: longest prime subsequence <=10000 with nondecreasing gaps via vectorized DP on pairs."""
import time
from math import isqrt
from bisect import bisect_left
import numpy as np

def construct():
    deadline = time.time() + 80
    N = 10000
    is_prime = bytearray([1]) * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, isqrt(N) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = 0
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    P = len(primes)
    if P < 2:
        return primes

    dp = np.full((P, P), 2, dtype=np.int16)
    prev = np.full((P, P), -1, dtype=np.int16)
    best_len = 2
    best_pair = (0, 1)
    primes_arr = np.array(primes, dtype=np.int32)
    
    for i in range(1, P):
        if time.time() > deadline - 15:
            break
        # build suffix max over k in [0, i-1] of dp[k][i]
        col_i = dp[:i, i]
        suff = np.maximum.accumulate(col_i[::-1])[::-1]
        
        # Vectorized update for j > i
        # T_j = 2 * primes[i] - primes[j]
        # k_min_j = lower_bound(primes[:i], T_j)
        # cand_j = 1 + suff[k_min_j] if k_min_j < i else 2
        
        T_arr = 2 * primes[i] - primes_arr[i+1:]
        # searchsorted on primes_arr[:i]
        k_min_arr = np.searchsorted(primes_arr[:i], T_arr, side='left')
        
        # suff is defined for indices 0..i-1.
        # If k_min_arr >= i, cand = 2. Else cand = 1 + suff[k_min_arr].
        
        # Create a safe index array for suff lookup
        # We need to handle k_min_arr >= i separately.
        valid_mask = k_min_arr < i
        safe_indices = np.where(valid_mask, k_min_arr, 0) # dummy index for invalid
        
        cand_arr = np.full(T_arr.shape, 2, dtype=np.int16)
        if np.any(valid_mask):
            cand_arr[valid_mask] = 1 + suff[safe_indices[valid_mask]]
        
        dp[i, i+1:] = cand_arr
        
        # Update best
        # Find max in dp[i, i+1:]
        if i + 1 < P:
            max_cand = np.max(cand_arr)
            if max_cand > best_len:
                best_len = int(max_cand)
                # Find index of max
                # We need the global index j
                local_idx = np.argmax(cand_arr)
                global_j = i + 1 + local_idx
                best_pair = (i, global_j)
                # Store prev for this pair? 
                # We need to store prev for ALL pairs to reconstruct.
                # prev[i, j] = k_min that gave the max?
                # Actually, we need the specific k that achieved the max value.
                # suff[k_min] is the max value. But which k achieved it?
                # We need to store the argmax of dp[k, i] for k in [k_min, i-1].
                # This is harder to vectorize fully without knowing the argmax.
                # However, for reconstruction, we can just search again as before, 
                # or we can store the argmax during the suffix computation.
                
                # Let's stick to the reconstruction search for simplicity and correctness,
                # but the vectorized DP makes the main loop faster.
                # The reconstruction is O(L * P) which is fine for L~100, P~1000.

    # Reconstruct sequence from best_pair
    if best_len <= 2:
        return primes[:best_len] if best_len >= 2 else primes
    
    seq = [primes[best_pair[1]], primes[best_pair[0]]]
    i, j = best_pair
    while len(seq) < best_len:
        target = int(dp[i, j]) - 1
        gap_ij = primes[j] - primes[i]
        found = False
        # Search for k
        # We can optimize this by using the fact that we know the max value is 'target'
        # and we need k >= k_min.
        T = 2 * primes[i] - primes[j]
        k_min = bisect_left(primes, T, 0, i)
        for k in range(i - 1, k_min - 1, -1):
            if int(dp[k, i]) == target:
                seq.append(primes[k])
                j, i = i, k
                found = True
                break
        if not found:
            # Fallback: search all k
            for k in range(i - 1, -1, -1):
                if int(dp[k, i]) == target and (primes[i] - primes[k]) <= gap_ij:
                    seq.append(primes[k])
                    j, i = i, k
                    found = True
                    break
        if not found:
            break
    seq.reverse()

    # Greedy extension
    if len(seq) >= 2 and time.time() < deadline - 10:
        last_gap = seq[-1] - seq[-2]
        last_val = seq[-1]
        idx = bisect_left(primes, last_val + last_gap)
        if idx < P:
            seq.append(primes[idx])
            while time.time() < deadline - 5:
                lg = seq[-1] - seq[-2]
                lv = seq[-1]
                nxt = bisect_left(primes, lv + lg)
                if nxt < P:
                    seq.append(primes[nxt])
                else:
                    break

    return seq
