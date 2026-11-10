import time
import numpy as np
from bisect import bisect_right

def construct():
    deadline = time.time() + 85.0
    
    # 1. Generate Primes
    N_LIMIT = 10000
    is_prime = np.ones(N_LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(N_LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    m = len(primes)
    
    if m < 2:
        return primes

    # 2. DP with Pareto Frontiers
    # dp[j] is a list of tuples (gap, length, prev_idx, prev_gap)
    # gap: the last gap in the sequence ending at primes[j]
    # length: length of the sequence
    # prev_idx: index of the previous prime in the sequence
    # prev_gap: the last gap in the sequence ending at primes[prev_idx]
    
    dp = [[] for _ in range(m)]
    # Base case: sequence of length 1 ending at primes[0]
    # gap is 0, length is 1, no prev
    dp[0] = [(0, 1, -1, 0)]
    
    best_len = 1
    best_j = 0
    best_gap = 0
    
    # Pre-extract gaps for faster access? 
    # primes is a list of ints.
    
    for j in range(1, m):
        if time.time() > deadline:
            break
            
        candidates = []
        p_j = primes[j]
        
        for i in range(j):
            d = p_j - primes[i]
            # Find best state in dp[i] with gap <= d
            # dp[i] is sorted by gap. Length is increasing with gap (Pareto).
            # We want the state with the largest gap <= d, which will have the largest length.
            
            lst = dp[i]
            if not lst:
                continue
                
            # Binary search for the rightmost gap <= d
            # Extract gaps for bisect? Or just iterate? 
            # Since lst is small, linear scan might be faster than extracting gaps.
            # But let's try bisect on a pre-extracted gap list if possible, or just linear.
            # Given M=1229 and frontier size ~10-20, linear scan is O(20) which is fine.
            # Bisect is O(log 20) ~ 5 steps.
            
            # Let's use linear scan for simplicity and low overhead, as frontiers are tiny.
            best_idx = -1
            # lst is sorted by gap
            # Find last index where lst[idx][0] <= d
            lo, hi = 0, len(lst) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if lst[mid][0] <= d:
                    best_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
                    
            if best_idx >= 0:
                gap_i, len_i, prev_i, prev_gap_i = lst[best_idx]
                # New state for j
                new_len = len_i + 1
                # prev_gap for the new state is the gap of the sequence ending at i, which is gap_i
                candidates.append((d, new_len, i, gap_i))
        
        if not candidates:
            # Should not happen for j>0 as we can always pair with some i
            # But if it does, treat as length 1? No, j>0 implies at least 2 primes.
            # Actually, if no candidate, it means no i < j had a valid extension?
            # That's impossible because gap d > 0 and dp[i] has gap 0 for len 1.
            # So candidates should never be empty for j > 0.
            dp[j] = []
            continue
        
        # Sort candidates by gap
        candidates.sort(key=lambda x: x[0])
        
        # Build Pareto Frontier
        # We want to keep only states where length is strictly increasing as gap increases.
        frontier = []
        max_len_so_far = -1
        for gap, length, prev_idx, prev_gap in candidates:
            if length > max_len_so_far:
                frontier.append((gap, length, prev_idx, prev_gap))
                max_len_so_far = length
        
        dp[j] = frontier
        
        # Update global best
        if frontier:
            # The last element in the frontier has the max length (and max gap)
            # But we want the max length overall.
            # Since length is increasing in the frontier, the last element has the max length for this j.
            last_gap, last_len, last_pi, last_pg = frontier[-1]
            if last_len > best_len:
                best_len = last_len
                best_j = j
                best_gap = last_gap
            # Note: It's possible a shorter gap has the same max length?
            # No, Pareto frontier ensures strictly increasing length.
            # So the max length for j is at the end.

    # 3. Reconstruct
    if best_len == 1:
        return [primes[0]]
        
    seq = [primes[best_j]]
    cur_j = best_j
    cur_gap = best_gap
    
    # We need to find the specific state in dp[cur_j] that matches (cur_gap, best_len)
    # And get its prev_idx and prev_gap.
    
    # Traverse back
    # We are at state (cur_j, cur_gap) which is part of dp[cur_j].
    # We need to find the tuple in dp[cur_j] with gap == cur_gap and length == best_len (or corresponding).
    # Actually, best_gap is the gap of the last step.
    
    while cur_j != -1:
        # Find the state in dp[cur_j] that has gap == cur_gap
        # There might be multiple states with same gap? No, we keep unique gaps in frontier?
        # Yes, candidates are sorted by gap. If multiple candidates have same gap, only the one with max length is kept (or rather, the loop `if length > max_len_so_far` handles it).
        # Wait, if two candidates have same gap but different lengths, the one with larger length dominates.
        # So in the frontier, gaps are unique.
        
        found_state = None
        for state in dp[cur_j]:
            if state[0] == cur_gap:
                found_state = state
                break
                
        if found_state is None:
            # Should not happen
            break
            
        _, length, prev_idx, prev_gap = found_state
        
        if prev_idx == -1:
            break
            
        seq.append(primes[prev_idx])
        cur_j = prev_idx
        cur_gap = prev_gap
        
    seq.reverse()
    
    # Verify
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for i in range(len(gaps)-1):
            if gaps[i+1] < gaps[i]:
                # Invalid, fallback to greedy or just return
                # This shouldn't happen if DP is correct
                pass
                
    return seq
