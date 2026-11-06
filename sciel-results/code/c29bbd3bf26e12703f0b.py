import time
import math
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin for 90s limit
    
    N_LIMIT = 10000
    
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N_LIMIT + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for p in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = bytearray(len(is_prime[p*p::p]))
            
    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)
    
    if n < 2:
        return primes
        
    # dp[i][j] = length of longest valid subsequence ending with primes[j], primes[i]
    # where j < i
    # We store prev[i][j] to reconstruct the path
    # Using list of lists for dp and prev
    # n is ~1229, n*n is ~1.5M. This fits in memory.
    
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    max_len = 0
    best_i = -1
    best_j = -1
    
    # Iterate over the second-to-last prime index j
    for j in range(n):
        if time.time() - t0 > DEADLINE:
            break
            
        if j == 0:
            continue
            
        # We will compute dp[i][j] for all i > j
        # We maintain the max of dp[j][k] for k in [k_min, j-1]
        # As i increases, k_min decreases, so the range expands to the left.
        
        curr_k_min = j # Initially no k is included (range [j, j-1] is empty)
        max_val = 0
        best_k = -1
        
        for i in range(j + 1, n):
            # Calculate the minimum prime value for k
            # p_k >= 2 * p_j - p_i
            limit = 2 * primes[j] - primes[i]
            
            # Find the smallest index k_min such that primes[k_min] >= limit
            # We only care about k < j
            k_min = bisect.bisect_left(primes, limit, 0, j)
            
            # Update max_val with new k's in [k_min, curr_k_min - 1]
            # Note: k_min <= curr_k_min
            # We iterate downwards from curr_k_min - 1 to k_min
            # This ensures we cover all new elements
            
            # Optimization: if k_min == curr_k_min, no new elements
            if k_min < curr_k_min:
                # Scan the new range
                # dp[j][k] is the value we are maximizing over
                row_j = dp[j]
                for k in range(curr_k_min - 1, k_min - 1, -1):
                    val = row_j[k]
                    if val > max_val:
                        max_val = val
                        best_k = k
                curr_k_min = k_min
            
            # Determine dp[i][j]
            if max_val > 0:
                length = max_val + 1
                dp[i][j] = length
                prev[i][j] = best_k
            else:
                length = 2
                dp[i][j] = 2
                prev[i][j] = -1
                
            if length > max_len:
                max_len = length
                best_i = i
                best_j = j

    # Reconstruct the sequence
    if max_len == 0:
        return primes[:1] if n > 0 else []
        
    seq = [primes[best_i], primes[best_j]]
    cur_i, cur_j = best_i, best_j
    
    while prev[cur_i][cur_j] != -1:
        k = prev[cur_i][cur_j]
        seq.append(primes[k])
        cur_i, cur_j = cur_j, k
        
    seq.reverse()
    
    return seq
