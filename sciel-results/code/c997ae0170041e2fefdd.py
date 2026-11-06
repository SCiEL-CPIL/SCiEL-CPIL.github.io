import math
import bisect

def construct():
    # Sieve of Eratosthenes to generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [primes[0]]
    
    # dp[i][j] = max length of valid sequence ending at primes[i] with previous prime primes[j]
    # prev[i][j] = index k such that primes[k] is the previous prime in the optimal sequence ending at primes[j] 
    #              that allows extending to primes[i]
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    # Iterate through potential "second to last" primes j
    for j in range(N):
        if j == 0:
            # No previous prime k < 0
            # Any i > 0 can form a sequence of length 2 with j=0
            for i in range(1, N):
                dp[i][j] = 2
                prev[i][j] = -1
        else:
            # Prepare data for predecessor k < j
            # We need (gap, dp[j][k], k) for all k < j
            items = []
            for k in range(j):
                gap = primes[j] - primes[k]
                length = dp[j][k]
                items.append((gap, length, k))
            
            # Sort by gap
            items.sort(key=lambda x: x[0])
            
            # Separate into lists for faster access and bisect
            sorted_gaps = [item[0] for item in items]
            sorted_lens = [item[1] for item in items]
            sorted_ks = [item[2] for item in items]
            
            # Build prefix maximums of length
            # prefix_max_lens[i] = max(sorted_lens[0..i])
            prefix_max_lens = [0] * len(items)
            current_max_len = 0
            for idx in range(len(items)):
                if sorted_lens[idx] > current_max_len:
                    current_max_len = sorted_lens[idx]
                prefix_max_lens[idx] = current_max_len
            
            # Iterate through potential "last" primes i > j
            for i in range(j + 1, N):
                gap_needed = primes[i] - primes[j]
                
                # Binary search for the largest gap <= gap_needed
                # bisect_right returns the insertion point to the right of any existing entries equal to gap_needed
                # We want the last index where gap <= gap_needed
                best_idx = bisect.bisect_right(sorted_gaps, gap_needed) - 1
                
                if best_idx != -1:
                    # We can extend a sequence ending at j
                    best_len = prefix_max_lens[best_idx]
                    best_k = sorted_ks[best_idx]
                    dp[i][j] = best_len + 1
                    prev[i][j] = best_k
                else:
                    # No valid predecessor k, so start new sequence of length 2
                    dp[i][j] = 2
                    prev[i][j] = -1

    # Find the maximum length and the ending indices
    max_len = 0
    end_i = -1
    end_j = -1
    
    for i in range(N):
        for j in range(i):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                end_i = i
                end_j = j

    # If no pair was found (N < 2), handle separately, but N>=2 is guaranteed here
    if end_i == -1:
        return [primes[0]]

    # Backtrack to reconstruct the sequence
    path = [primes[end_i], primes[end_j]]
    i = end_i
    j = end_j
    
    while prev[i][j] != -1:
        k = prev[i][j]
        path.append(primes[k])
        i, j = j, k
        
    path.reverse()
    return path
