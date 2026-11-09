"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 10000
    # Sieve of Eratosthenes to get primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    n = len(primes)
    
    if n <= 1:
        return primes.tolist()
    
    # Compute gaps between consecutive primes in the full list
    # But we need to consider arbitrary subsequences, so we need DP over all pairs
    
    # dp[i] = length of longest valid subsequence ending at primes[i]
    # gap[i] = the last gap used to get to primes[i]
    # We need to track the last gap to enforce nondecreasing gaps
    
    # State: dp[i][g] would be too large. Instead, we can use:
    # For each prime i, store the best (max length) for each possible last gap.
    # But gaps can be up to ~200 or so for primes <= 10000.
    
    # Actually, let's think differently. The gap between two primes p_i and p_j (i < j) is p_j - p_i.
    # We want a subsequence p_{i1} < p_{i2} < ... < p_{ik} such that
    # p_{i2} - p_{i1} <= p_{i3} - p_{i2} <= ... <= p_{ik} - p_{i{k-1}}
    
    # DP: Let f(j, g) = max length of valid subsequence ending at prime j with last gap exactly g.
    # But g can be large. However, the number of distinct gaps is limited.
    
    # Alternative: For each prime j, let best[j] = max over all valid subsequences ending at j of the length.
    # But we also need to know the last gap. So we need a 2D state.
    
    # Let's use: dp[j] = dict mapping last_gap -> max_length
    # This might be too slow in Python. Let's try NumPy vectorization.
    
    # Actually, the max gap between primes <= 10000 is at most ~200 (the maximal prime gap below 10000 is around 114 or so).
    # Let's find the max possible gap.
    max_gap = int(np.max(np.diff(primes))) if n > 1 else 0
    # But gaps in subsequence can be larger than consecutive prime gaps. E.g., p_j - p_i for i << j.
    # Max gap is primes[-1] - primes[0] = 10000 - 2 = 9998. That's too large for a 2D array.
    
    # Let's use a different approach: 
    # dp[i] = (max_length, min_last_gap_for_that_length) ... no, that's not sufficient.
    
    # Let's just do O(n^2) with Python loops but optimized. n = 1229 primes, so n^2 ~ 1.5M, which is fine.
    
    # dp[i] = list of (last_gap, length) pairs, where for each last_gap we keep the max length.
    # Actually, for each prime i, we only need to keep the Pareto-optimal (gap, length) pairs.
    # A pair (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2.
    
    # Let's use a simpler approach: for each prime i, store an array best_gap where best_gap[g] = max length ending at i with last gap g.
    # But g can be up to ~10000, so this is a 1229 x 10000 array, which is 12M entries. That's fine in NumPy.
    
    # Actually, let's just use a 2D NumPy array: dp[i, g] = max length of subsequence ending at prime i with last gap g.
    # Initialize dp[i, 0] = 1 for all i (subsequence of just prime i, no gap).
    # Transition: for i < j, gap = primes[j] - primes[i].
    #   dp[j, gap] = max(dp[j, gap], max over g' <= gap of dp[i, g'] + 1)
    # This is O(n^2 * max_gap) which is too slow.
    
    # Let's go back to the simple O(n^2) Python approach with a clever state.
    # For each prime i, maintain a list of (gap, length) that are Pareto-optimal.
    
    # Actually, let's just try the straightforward O(n^2) with a dict per prime.
    
    from collections import defaultdict
    
    # dp[i] = dict: last_gap -> max_length
    # For prime 0 (the first prime), dp[0] = {0: 1} (gap 0 means no previous prime)
    
    dp = [dict() for _ in range(n)]
    dp[0][0] = 1  # Special: gap 0 means starting here
    
    for j in range(1, n):
        for i in range(j):
            gap = primes[j] - primes[i]
            # For each (g', l') in dp[i], if g' <= gap, we can extend
            # We want to find the max l' among all g' <= gap
            best_l = 0
            for g_prime, l_prime in dp[i].items():
                if g_prime <= gap:
                    if l_prime > best_l:
                        best_l = l_prime
            if best_l > 0:
                new_l = best_l + 1
                if new_l > dp[j].get(gap, 0):
                    dp[j][gap] = new_l
    
    # Find the max length and reconstruct
    max_len = 0
    best_j = -1
    best_gap = -1
    for j in range(n):
        for gap, l in dp[j].items():
            if l > max_len:
                max_len = l
                best_j = j
                best_gap = gap
    
    # Reconstruct the sequence
    result = [primes[best_j]]
    current_j = best_j
    current_gap = best_gap
    while current_gap > 0:
        # Find i such that primes[current_j] - primes[i] == current_gap and dp[i] has a state with gap <= current_gap giving length max_len - 1
        # Actually, we need to trace back properly.
        # Let's store parent pointers instead.
        # This reconstruction is getting complex. Let's just return the length for now... 
        # No, we need to return the actual primes.
        
        # Let's redo with parent tracking.
        break
    
    # Let's redo with parent tracking from the start.
    dp = [dict() for _ in range(n)]
    parent = [dict() for _ in range(n)]  # parent[i][gap] = (prev_prime_index, prev_gap)
    
    dp[0][0] = 1
    
    for j in range(1, n):
        for i in range(j):
            gap = primes[j] - primes[i]
            best_l = 0
            best_prev_gap = -1
            for g_prime, l_prime in dp[i].items():
                if g_prime <= gap and l_prime > best_l:
                    best_l = l_prime
                    best_prev_gap = g_prime
            if best_l > 0:
                new_l = best_l + 1
                if new_l > dp[j].get(gap, 0):
                    dp[j][gap] = new_l
                    parent[j][gap] = (i, best_prev_gap)
    
    max_len = 0
    best_j = -1
    best_gap = -1
    for j in range(n):
        for gap, l in dp[j].items():
            if l > max_len:
                max_len = l
                best_j = j
                best_gap = gap
    
    # Reconstruct
    result = [primes[best_j]]
    cur_j = best_j
    cur_gap = best_gap
    while cur_gap > 0:
        prev_j, prev_gap = parent[cur_j][cur_gap]
        result.append(primes[prev_j])
        cur_j = prev_j
        cur_gap = prev_gap
    
    result.reverse()
    return result
# EVOLVE-BLOCK-END
