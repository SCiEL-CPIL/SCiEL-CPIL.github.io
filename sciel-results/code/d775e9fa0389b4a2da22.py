def construct():
    from math import isqrt
    
    # Generate primes up to 10000 using sieve
    sieve = [True] * 10001
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(10000) + 1):
        if sieve[i]:
            for j in range(i * i, 10001, i):
                sieve[j] = False
    primes = [i for i in range(2, 10001) if sieve[i]]
    n = len(primes)
    
    if n <= 1:
        return primes
    
    # dp[i][j] = length of longest valid sequence ending with primes[j], primes[i] (j < i)
    dp = [[0] * n for _ in range(n)]
    suf = [None] * n
    
    best_len = 1
    best_i, best_j = 0, -1
    
    for i in range(n):
        for j in range(i):
            gap = primes[i] - primes[j]
            if j == 0:
                dp[i][j] = 2
            else:
                target = primes[j] - gap
                # Binary search for smallest k in [0, j) with primes[k] >= target
                lo, hi = 0, j
                while lo < hi:
                    mid = (lo + hi) // 2
                    if primes[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < j:
                    best_prev = suf[j][lo]
                    dp[i][j] = 2 + best_prev if best_prev > 0 else 2
                else:
                    dp[i][j] = 2
            
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i, best_j = i, j
        
        # Build suffix max for row i
        if i > 0:
            suf_arr = [0] * i
            suf_arr[i - 1] = dp[i][i - 1]
            for k in range(i - 2, -1, -1):
                suf_arr[k] = max(dp[i][k], suf_arr[k + 1])
            suf[i] = suf_arr
    
    # Reconstruct sequence
    if best_j == -1:
        return [primes[0]]
    
    seq = [primes[best_i], primes[best_j]]
    cur_i, cur_j = best_i, best_j
    while dp[cur_i][cur_j] > 2:
        needed = dp[cur_i][cur_j] - 2
        gap = primes[cur_i] - primes[cur_j]
        target = primes[cur_j] - gap
        lo, hi = 0, cur_j
        while lo < hi:
            mid = (lo + hi) // 2
            if primes[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        found = False
        for k in range(lo, cur_j):
            if dp[cur_j][k] == needed:
                seq.append(primes[k])
                cur_i, cur_j = cur_j, k
                found = True
                break
        if not found:
            break
    
    seq.reverse()
    return seq
