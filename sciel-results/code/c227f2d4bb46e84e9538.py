"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Generate primes up to 10000 using sieve
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n <= 1:
        return primes
    
    # dp[j] = length of longest valid subsequence ending at primes[j]
    # prev[j] = index of previous prime in the optimal subsequence ending at primes[j]
    dp = [1] * n
    prev = [-1] * n
    
    # For each prime j, we want to find i < j such that:
    #   primes[j] - primes[i] >= gap_before_i_in_its_subsequence
    # where gap_before_i is the gap between prev[i] and i (if prev[i] exists)
    # 
    # We process primes in order. For each j, we need to query over all i < j
    # where the "required min gap" (gap from prev[i] to i) is <= (primes[j] - primes[i]).
    #
    # Alternative: For each pair (i, j) with i < j, if primes[j] - primes[i] >= 
    # (primes[i] - primes[prev[i]]) when prev[i] >= 0, or always valid if prev[i] == -1,
    # then dp[j] = max(dp[j], dp[i] + 1)
    #
    # This is O(n^2) which is ~1228^2 ≈ 1.5M operations - feasible.
    
    # For each j, iterate i from j-1 down to 0
    # We need: gap_ji = primes[j] - primes[i]
    #          required = primes[i] - primes[prev[i]] if prev[i] >= 0 else 0
    #          if gap_ji >= required: dp[j] = max(dp[j], dp[i] + 1)
    
    for j in range(1, n):
        best_i = -1
        best_len = 1
        for i in range(j):
            gap_ji = primes[j] - primes[i]
            if prev[i] >= 0:
                required = primes[i] - primes[prev[i]]
            else:
                required = 0
            if gap_ji >= required:
                if dp[i] + 1 > best_len:
                    best_len = dp[i] + 1
                    best_i = i
        if best_i >= 0:
            dp[j] = best_len
            prev[j] = best_i
    
    # Find the longest subsequence
    max_len = max(dp)
    end_idx = dp.index(max_len)
    
    # Reconstruct
    result = []
    idx = end_idx
    while idx >= 0:
        result.append(primes[idx])
        idx = prev[idx]
    result.reverse()
    
    return result
# EVOLVE-BLOCK-END
