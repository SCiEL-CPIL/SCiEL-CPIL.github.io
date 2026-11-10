import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    # 2. Dynamic Programming
    # dp[i]: length of longest valid subsequence ending at primes[i]
    # prev[i]: index of previous prime in the subsequence
    # last_gap[i]: gap between primes[i] and primes[prev[i]] (0 if length 1)
    
    dp = [1] * n
    prev = [-1] * n
    last_gap = [0] * n
    
    best_len = 1
    best_idx = 0
    
    for i in range(1, n):
        pi = primes[i]
        for j in range(i):
            pj = primes[j]
            gap = pi - pj
            
            # Check if we can extend the subsequence ending at j
            # If dp[j] == 1, there is no previous gap constraint (or gap >= 0)
            # Otherwise, new gap must be >= last_gap[j]
            if dp[j] == 1 or gap >= last_gap[j]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j
                    last_gap[i] = gap
                    
                    if dp[i] > best_len:
                        best_len = dp[i]
                        best_idx = i
                        
        # Time check (optional, but good practice)
        if time.time() > deadline:
            break
            
    # 3. Reconstruct the sequence
    seq = []
    curr = best_idx
    while curr != -1:
        seq.append(primes[curr])
        curr = prev[curr]
        
    seq.reverse()
    return seq
