"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    import bisect

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(10000)
    n = len(primes)
    
    if n <= 2:
        return primes

    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # parent[i][j] = h such that primes[h] is the predecessor of primes[i] in the optimal path
    dp = [[2] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    best_len = 2
    best_pair = (0, 1)

    # Iterate through the second-to-last element index i
    for i in range(1, n - 1):
        # Collect potential predecessors h < i
        # We need (gap, dp[h][i], h)
        candidates = []
        for h in range(i):
            gap = primes[i] - primes[h]
            candidates.append((gap, dp[h][i], h))
        
        # Sort by gap ascending
        candidates.sort(key=lambda x: x[0])
        
        # Prepare prefix max structure for O(1) lookup after binary search
        gaps = [c[0] for c in candidates]
        prefix_max_len = [0] * len(candidates)
        prefix_max_idx = [0] * len(candidates)
        
        prefix_max_len[0] = candidates[0][1]
        prefix_max_idx[0] = candidates[0][2]
        
        for k in range(1, len(candidates)):
            if candidates[k][1] > prefix_max_len[k - 1]:
                prefix_max_len[k] = candidates[k][1]
                prefix_max_idx[k] = candidates[k][2]
            else:
                prefix_max_len[k] = prefix_max_len[k - 1]
                prefix_max_idx[k] = prefix_max_idx[k - 1]

        # Iterate through the last element index j > i
        for j in range(i + 1, n):
            required_gap = primes[j] - primes[i]
            # Find the best h such that gap_h <= required_gap
            # bisect_right gives the index of the first element > required_gap
            # We want the index of the last element <= required_gap
            idx = bisect.bisect_right(gaps, required_gap) - 1
            
            if idx >= 0:
                new_len = 2 + prefix_max_len[idx]
                dp[i][j] = new_len
                parent[i][j] = prefix_max_idx[idx]
                
                if new_len > best_len:
                    best_len = new_len
                    best_pair = (i, j)
            # else: dp[i][j] remains 2, parent remains -1

    # Backtrack to reconstruct the sequence
    seq = []
    i, j = best_pair
    while True:
        seq.append(primes[j])
        h = parent[i][j]
        if h == -1:
            break
        j = i
        i = h
    seq.append(primes[i])
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END
