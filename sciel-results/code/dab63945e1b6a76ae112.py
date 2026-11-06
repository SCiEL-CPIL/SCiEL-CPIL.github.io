import time
from math import isqrt
from bisect import bisect_left

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve of Eratosthenes
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    num_primes = len(primes)
    
    # DP approach: dp[i] = length of longest valid subsequence ending at primes[i]
    # prev[i] = previous index in the optimal subsequence ending at primes[i]
    # For each i, we look at all j < i where primes[i] - primes[j] >= (gap from j-1 to j)
    # Actually, we need to track the last gap as well.
    
    # Better DP: dp[i][g] is not practical. Instead:
    # Let dp[i] = (length, last_gap, prev_index) for the best subsequence ending at primes[i]
    # But "best" is ambiguous - we want maximum length.
    # 
    # Actually, the standard approach:
    # dp[i] = length of longest valid nondecreasing-gap subsequence ending at primes[i]
    # To compute dp[i], for each j < i, if primes[i] - primes[j] >= gap_j (where gap_j is the gap
    # used in the subsequence ending at j), then dp[i] = max(dp[i], dp[j] + 1)
    # But we need to know the gap used at j. So we need more state.
    #
    # Alternative: dp[i] = list of (last_gap, length) pairs, where for each possible last_gap,
    # we store the maximum length. But this could be large.
    #
    # Simpler approach that works well:
    # For each i, dp[i] = max over j < i of (dp[j] + 1) where primes[i] - primes[j] >= gap_at_j
    # We need to track gap_at_j. Let's use:
    # dp[i] = (best_length, last_gap_used, prev_idx)
    # But multiple states may be valid for the same i with different last gaps.
    #
    # Let's use: for each i, maintain a list of (last_gap, length) where length is the max
    # length achievable ending at i with that last_gap. Then for each new i, we check all j < i.
    # This is O(n^2) in the number of primes, which is ~1229^2 ≈ 1.5M, very feasible.
    
    # dp[i] = list of (last_gap, length) tuples
    # Initialize: dp[i] = [(0, 1)] for each i (single element, gap 0)
    # Transition: for i > j, for each (gap_j, len_j) in dp[j]:
    #   new_gap = primes[i] - primes[j]
    #   if new_gap >= gap_j:
    #     dp[i] can have (new_gap, len_j + 1)
    
    # To keep dp[i] manageable, for each i we only keep Pareto-optimal (gap, length) pairs:
    # a pair (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2.
    
    dp = [[] for _ in range(num_primes)]
    for i in range(num_primes):
        dp[i] = [(0, 1)]
    
    # Track best
    best_length = 1
    best_end = 0
    
    for i in range(1, num_primes):
        for j in range(i):
            gap_j_prime = primes[i] - primes[j]
            for (last_gap, length) in dp[j]:
                if gap_j_prime >= last_gap:
                    new_length = length + 1
                    # Try to add (gap_j_prime, new_length) to dp[i]
                    # Check if it's dominated by existing entries
                    dominated = False
                    for (g, l) in dp[i]:
                        if g <= gap_j_prime and l >= new_length:
                            dominated = True
                            break
                    if not dominated:
                        # Remove entries dominated by the new one
                        dp[i] = [(g, l) for (g, l) in dp[i] if not (gap_j_prime <= g and new_length >= l)]
                        dp[i].append((gap_j_prime, new_length))
        
        # Update best
        for (g, l) in dp[i]:
            if l > best_length:
                best_length = l
                best_end = i
    
    # Reconstruct the best sequence
    # We need to track the actual path. Let's redo with path tracking.
    # parent[i] = (prev_idx, gap_used) for the best path to i
    # But we have multiple states per i. Let's track for the best state at each i.
    
    # Actually, let's just track the best length and reconstruct greedily.
    # Or better: store for each i and each (gap, length) state, the predecessor.
    
    # Simpler: store best_len[i] and best_prev[i] and best_gap[i] for the single best state.
    # But that might miss optimal paths. Let's use the full DP with path info.
    
    # Let me redo: dp[i] is a list of (last_gap, length, prev_idx, prev_gap_in_prev_state)
    # This is getting complex. Let me just do O(n^2) with proper tracking.
    
    # Reset
    dp = [[] for _ in range(num_primes)]
    for i in range(num_primes):
        dp[i] = [(0, 1, -1, 0)]  # (last_gap, length, prev_idx, prev_state_gap)
    
    best_length = 1
    best_end = 0
    best_gap_at_end = 0
    
    for i in range(1, num_primes):
        for j in range(i):
            gap_ij = primes[i] - primes[j]
            for (last_gap, length, prev_idx, prev_gap) in dp[j]:
                if gap_ij >= last_gap:
                    new_length = length + 1
                    # Check if dominated
                    dominated = False
                    for (g, l, pi, pg) in dp[i]:
                        if g <= gap_ij and l >= new_length:
                            dominated = True
                            break
                    if not dominated:
                        dp[i] = [(g, l, pi, pg) for (g, l, pi, pg) in dp[i] 
                                  if not (gap_ij <= g and new_length >= l)]
                        dp[i].append((gap_ij, new_length, j, last_gap))
        
        for (g, l, pi, pg) in dp[i]:
            if l > best_length:
                best_length = l
                best_end = i
                best_gap_at_end = g
    
    # Reconstruct
    sequence = []
    cur_idx = best_end
    cur_gap = best_gap_at_end
    while cur_idx != -1:
        sequence.append(primes[cur_idx])
        # Find the state in dp[cur_idx] with last_gap = cur_gap
        prev_idx = -1
        for (g, l, pi, pg) in dp[cur_idx]:
            if g == cur_gap:
                prev_idx = pi
                cur_gap = pg
                break
        cur_idx = prev_idx
    
    sequence.reverse()
    return sequence
