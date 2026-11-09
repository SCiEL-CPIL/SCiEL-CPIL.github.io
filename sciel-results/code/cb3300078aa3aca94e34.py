import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    limit = 10000

    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    if n < 2:
        return primes

    # dp[j] will be a list of length j.
    # dp[j][i] = length of longest non-decreasing gap sequence ending with primes[i], primes[j]
    # where i < j.
    # parent[j] will be a list of length j.
    # parent[j][i] = k < i such that dp[j][i] = dp[i][k] + 1. If no such k, parent is -1.
    
    dp = [None] * n
    parent = [None] * n
    
    # Initialize for length 2 sequences
    # For any pair (i, j) with i < j, the sequence [p_i, p_j] is valid (one gap).
    # So dp[j][i] = 2 initially.
    # We will update it if we can extend it.
    
    # Actually, it's easier to initialize dp[j] as a list of 2s for all i < j,
    # but we only compute dp[j][i] when processing i.
    # Let's pre-allocate.
    for j in range(n):
        dp[j] = [2] * j
        parent[j] = [-1] * j

    best_len = 2
    best_i = 0
    best_j = 1

    # Iterate i from 1 to n-2
    # For each i, we have dp[i][k] for all k < i already computed.
    # We want to update dp[j][i] for all j > i.
    
    for i in range(1, n - 1):
        if time.time() > deadline:
            break

        # Build suffix max array for dp[i]
        # suffix_max[k] = max(dp[i][k], dp[i][k+1], ..., dp[i][i-1])
        # We also need to track the index that achieves the max for reconstruction.
        # Let's store (max_val, argmax_idx) in suffix arrays.
        
        suffix_max = [0] * i
        suffix_argmax = [-1] * i
        
        # Initialize for k = i-1
        suffix_max[i-1] = dp[i][i-1]
        suffix_argmax[i-1] = i-1
        
        for k in range(i-2, -1, -1):
            if dp[i][k] > suffix_max[k+1]:
                suffix_max[k] = dp[i][k]
                suffix_argmax[k] = k
            else:
                suffix_max[k] = suffix_max[k+1]
                suffix_argmax[k] = suffix_argmax[k+1]

        # For each j > i, compute dp[j][i]
        for j in range(i + 1, n):
            if time.time() > deadline:
                break

            # Condition: primes[i] - primes[k] <= primes[j] - primes[i]
            # => primes[k] >= 2*primes[i] - primes[j]
            threshold = 2 * primes[i] - primes[j]
            
            # Find smallest k in [0, i-1] such that primes[k] >= threshold
            # If no such k, then no valid predecessor.
            # If threshold <= primes[0], then k=0 is the smallest.
            # If threshold > primes[i-1], then no k works.
            
            # Binary search for leftmost index in primes[0:i] with value >= threshold
            lo, hi = 0, i - 1
            pos = i # Default: not found
            if primes[0] >= threshold:
                pos = 0
            elif primes[i-1] < threshold:
                pos = i # Not found
            else:
                while lo < hi:
                    mid = (lo + hi) // 2
                    if primes[mid] >= threshold:
                        hi = mid
                    else:
                        lo = mid + 1
                pos = lo

            if pos < i:
                # Valid k's are in [pos, i-1]
                # The max length is suffix_max[pos]
                new_len = suffix_max[pos] + 1
                if new_len > dp[j][i]:
                    dp[j][i] = new_len
                    parent[j][i] = suffix_argmax[pos]
                    if new_len > best_len:
                        best_len = new_len
                        best_i = i
                        best_j = j

    if time.time() > deadline:
        # Fallback to greedy if DP didn't finish (should not happen for n~1229)
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    # Reconstruct the sequence
    if best_len <= 2:
        # If no longer sequence found, return a pair or greedy
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    # Backtrack from (best_i, best_j)
    # Current pair is (best_i, best_j)
    seq = [primes[best_j], primes[best_i]]
    i_cur = best_i
    j_cur = best_j
    
    while parent[j_cur][i_cur] != -1:
        k = parent[j_cur][i_cur]
        seq.append(primes[k])
        j_cur = i_cur
        i_cur = k
        
    seq.reverse()
    return seq
