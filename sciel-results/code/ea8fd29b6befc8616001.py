import time
import math

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin

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
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    max_len = 0
    best_i = -1
    best_j = -1
    
    for j in range(n):
        if j == 0:
            for i in range(1, n):
                dp[i][0] = 2
                prev[i][0] = -1
                if dp[i][0] > max_len:
                    max_len = dp[i][0]
                    best_i = i
                    best_j = 0
        else:
            size = j
            # Build sparse table for array A where A[k] = (dp[j][k], k) for k in 0..j-1
            # We want max value, and if tie, smallest index (or any).
            # Let's store (value, index) and compare by value, then index.
            
            K = 1
            while (1 << K) <= size:
                K += 1
            # st[p][k] = (max_val, index) for range [k, k + 2^p - 1]
            st = [[(0, -1)] * size for _ in range(K)]
            
            # Base case
            for k in range(size):
                st[0][k] = (dp[j][k], k)
                
            for p in range(1, K):
                step = 1 << (p-1)
                for k in range(size - (1 << p) + 1):
                    left = st[p-1][k]
                    right = st[p-1][k + step]
                    if left[0] > right[0] or (left[0] == right[0] and left[1] < right[1]):
                        st[p][k] = left
                    else:
                        st[p][k] = right
            
            # rmq(l, r) returns (max_val, index)
            def rmq(l, r):
                if l > r:
                    return (0, -1)
                length = r - l + 1
                k = length.bit_length() - 1
                left = st[k][l]
                right = st[k][r - (1 << k) + 1]
                if left[0] > right[0] or (left[0] == right[0] and left[1] < right[1]):
                    return left
                else:
                    return right

            for i in range(j + 1, n):
                gap = primes[i] - primes[j]
                target = primes[j] - gap
                # Binary search in primes[0..j-1] for smallest k with primes[k] >= target
                lo, hi = 0, j - 1
                k_min = j
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if primes[mid] >= target:
                        k_min = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                
                if k_min < j:
                    max_val, best_k = rmq(k_min, j - 1)
                    dp[i][j] = 1 + max_val
                    prev[i][j] = best_k
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
                    
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j
                    
        if time.time() - t0 > DEADLINE:
            break

    if best_i == -1:
        return primes[:1]
        
    seq = [primes[best_i], primes[best_j]]
    cur_i = best_i
    cur_j = best_j
    while prev[cur_i][cur_j] != -1:
        cur_k = prev[cur_i][cur_j]
        seq.append(primes[cur_k])
        cur_i, cur_j = cur_j, cur_k
        
    seq.reverse()
    return seq
