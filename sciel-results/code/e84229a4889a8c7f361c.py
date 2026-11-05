def construct():
    import time
    start_time = time.time()
    DEADLINE = 85
    
    from math import isqrt
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [p for p in range(2, n+1) if is_prime[p]]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n <= 1:
        return primes
    
    # dp[i] is a list of (last_gap, length, prev_index, prev_state_idx)
    # sorted by last_gap, Pareto optimal
    dp = [[] for _ in range(n)]
    dp[0].append((0, 1, -1, -1))
    
    best_len = 1
    best_end = 0
    best_state_idx = 0
    
    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break
            
        candidates = [(0, 1, -1, -1)]
        
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            sj = dp[j]
            if not sj:
                continue
            
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            
            if lo > 0:
                prev_idx = lo - 1
                last_gap_j, len_j, _, _ = sj[prev_idx]
                candidates.append((gap, len_j + 1, j, prev_idx))
        
        if not candidates:
            dp[i] = []
            continue
            
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        filtered = []
        max_len_so_far = 0
        for gap, length, prev, p_idx in candidates:
            if length > max_len_so_far:
                filtered.append((gap, length, prev, p_idx))
                max_len_so_far = length
        
        dp[i] = filtered
        
        if filtered:
            cur_max_len = filtered[-1][1]
            if cur_max_len > best_len:
                best_len = cur_max_len
                best_end = i
                best_state_idx = len(filtered) - 1
                
    # Reconstruct
    if best_len <= 1:
        return [primes[0]]
        
    seq = []
    idx = best_end
    s_idx = best_state_idx
    while idx != -1:
        seq.append(primes[idx])
        gap, length, prev_idx, prev_s_idx = dp[idx][s_idx]
        idx = prev_idx
        s_idx = prev_s_idx
        
    seq.reverse()
    return seq
