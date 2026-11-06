"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    # 1. Generate primes <= 10000
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
    
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    n = len(primes)
    
    if n < 2:
        return primes
    
    # 2. DP for Longest Non-Decreasing Gap Sequence
    # dp[i] is a list of (last_gap, length) pairs.
    # We maintain the Pareto frontier: for a given length, we want the smallest last_gap.
    # Or for a given last_gap, we want the largest length.
    # Let's store (last_gap, length) and prune dominated pairs.
    # A pair (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2.
    
    dp = [[] for _ in range(n)]
    
    # Initialize: each prime starts a sequence of length 1 with gap 0 (or None)
    # But for extension, we need the last gap.
    # Let's define state as (last_gap, length).
    # For i=0, dp[0] = [(0, 1)]
    dp[0].append((0, 1))
    
    best_len = 1
    best_end_idx = 0
    best_last_gap = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        p_i = primes[i]
        current_dp = []
        
        # Start new sequence at i
        current_dp.append((0, 1))
        
        for j in range(i):
            if time.time() > deadline:
                break
                
            p_j = primes[j]
            gap = p_i - p_j
            
            # Try to extend sequences ending at j
            # We need a sequence ending at j with last_gap <= gap
            # We want the one with max length.
            # dp[j] is a list of (last_gap, length).
            # We can sort dp[j] by last_gap and binary search for the largest length with last_gap <= gap.
            # However, dp[j] changes as we process i? No, dp[j] is fixed for j < i.
            # But we need to optimize the lookup.
            
            # For efficiency, we can pre-process dp[j] to be sorted by last_gap.
            # Since we are iterating i from 0 to n-1, dp[j] is complete when we reach i > j.
            
            # Let's maintain dp[j] as a list of (last_gap, max_length_for_this_gap_or_less).
            # Actually, just iterating over dp[j] might be fast enough if N is small.
            # N ~ 1229. N^2 ~ 1.5M.
            # Inside, we iterate over dp[j]. If dp[j] is small, it's fast.
            
            for last_gap, length in dp[j]:
                if last_gap <= gap:
                    new_length = length + 1
                    current_dp.append((gap, new_length))
        
        # Prune current_dp to Pareto frontier
        # Sort by last_gap, then for same last_gap keep max length.
        # Then keep only pairs where length is strictly increasing as last_gap increases.
        if current_dp:
            current_dp.sort(key=lambda x: (x[0], -x[1]))
            pruned = []
            max_len_so_far = 0
            # We want to keep (g, l) if l > max_len_so_far for all g' < g.
            # Since sorted by g, we can just scan.
            # But we also want to minimize g for a given l?
            # Standard Pareto: (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2.
            # So we want to keep the "best" (smallest g) for each l, or "best" (largest l) for each g.
            
            # Let's group by length and keep min gap?
            # Or group by gap and keep max length?
            
            # Simpler:
            # Sort by gap.
            # Iterate. Keep a record of max length seen so far.
            # If current length > max_len_so_far, it's a new frontier point.
            # Wait, if I have (gap=10, len=5) and (gap=20, len=4), the second is dominated.
            # If I have (gap=10, len=4) and (gap=20, len=5), both are useful.
            
            frontier = []
            max_len = 0
            last_gap_seen = -1
            for g, l in current_dp:
                if l > max_len:
                    frontier.append((g, l))
                    max_len = l
                    if l > best_len:
                        best_len = l
                        best_end_idx = i
                        best_last_gap = g
                # If l <= max_len, it is dominated by some previous (g', l') with g' <= g and l' >= l.
                # Note: current_dp is sorted by g ascending.
        
            dp[i] = frontier
    
    # Reconstruct the sequence
    # We need to trace back.
    # This is hard with the pruned DP because we lost the specific j.
    # We need to store parent pointers.
    
    # Restart with Parent Pointers
    dp = [[] for _ in range(n)]
    parent = [None] * n  # parent[i] = (j, last_gap_used)
    
    dp[0].append((0, 1))
    best_len = 1
    best_end_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        p_i = primes[i]
        current_dp = []
        best_pair_for_i = None
        best_parent_for_i = None
        best_len_for_i = 0
        
        # Start new
        current_dp.append((0, 1))
        if 1 > best_len_for_i:
            best_len_for_i = 1
            best_pair_for_i = (0, 1)
            best_parent_for_i = None # start
        
        for j in range(i):
            if time.time() > deadline:
                break
            p_j = primes[j]
            gap = p_i - p_j
            
            for last_gap, length in dp[j]:
                if last_gap <= gap:
                    new_length = length + 1
                    # Check if this is better than current best for i
                    # We want to update the frontier of i.
                    # But for reconstruction, we just need the global best.
                    if new_length > best_len_for_i:
                        best_len_for_i = new_length
                        best_pair_for_i = (gap, new_length)
                        best_parent_for_i = (j, last_gap)
                    
                    current_dp.append((gap, new_length))
        
        # Prune current_dp for dp[i]
        if current_dp:
            current_dp.sort(key=lambda x: (x[0], -x[1]))
            frontier = []
            max_len = 0
            for g, l in current_dp:
                if l > max_len:
                    frontier.append((g, l))
                    max_len = l
            
            dp[i] = frontier
            parent[i] = best_parent_for_i
            
            if best_len_for_i > best_len:
                best_len = best_len_for_i
                best_end_idx = i
    
    # Reconstruct
    seq = []
    curr = best_end_idx
    while curr is not None:
        seq.append(primes[curr])
        if parent[curr] is None:
            break
        curr = parent[curr][0]
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
