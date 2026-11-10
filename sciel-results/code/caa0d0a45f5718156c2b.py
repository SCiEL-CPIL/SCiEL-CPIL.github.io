def construct():
    import time
    start_time = time.time()
    limit = 10000
    
    # Sieve
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    dp = [1] * N
    prev = [-1] * N
    
    # DP
    for i in range(1, N):
        pi = primes[i]
        best_len = 1
        best_prev = -1
        for j in range(i):
            pj = primes[j]
            gap_new = pi - pj
            if prev[j] == -1:
                gap_old = 0 # Effectively -inf, so any gap >= 0 works
            else:
                gap_old = pj - primes[prev[j]]
            
            if gap_new >= gap_old:
                if dp[j] + 1 > best_len:
                    best_len = dp[j] + 1
                    best_prev = j
        dp[i] = best_len
        prev[i] = best_prev
        
    # Find max
    max_len = max(dp)
    end_idx = dp.index(max_len)
    
    # Reconstruct
    path = []
    curr = end_idx
    while curr != -1:
        path.append(primes[curr])
        curr = prev[curr]
    path.reverse()
    
    return path
