import time
import math
import bisect

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
    
    # dp[i] = length of longest valid subsequence ending at primes[i]
    # prev[i] = index of predecessor of primes[i] in the subsequence (-1 if none)
    # last_gap[i] = the gap between primes[i] and its predecessor (0 if i is first)
    dp = [1] * n
    prev = [-1] * n
    last_gap = [0] * n  # 0 means "no previous gap" (first element)
    
    max_len = 1
    best_end = 0
    
    for i in range(1, n):
        best_dp_val = 1
        best_prev = -1
        best_lg = 0
        
        for j in range(i):
            gap = primes[i] - primes[j]
            # Can we extend the subsequence ending at j to include i?
            # Need: gap >= last_gap[j]
            if gap >= last_gap[j]:
                if dp[j] > best_dp_val:
                    best_dp_val = dp[j]
                    best_prev = j
                    best_lg = gap
        
        if best_prev == -1:
            dp[i] = 1
            prev[i] = -1
            last_gap[i] = 0
        else:
            dp[i] = best_dp_val + 1
            prev[i] = best_prev
            last_gap[i] = best_lg
        
        if dp[i] > max_len:
            max_len = dp[i]
            best_end = i
        
        if time.time() - t0 > DEADLINE:
            break
    
    if best_end == 0 and n > 1:
        # Try to at least get a pair
        return primes[:2]
    
    # Reconstruct
    seq = []
    cur = best_end
    while cur != -1:
        seq.append(primes[cur])
        cur = prev[cur]
    seq.reverse()
    
    return seq
