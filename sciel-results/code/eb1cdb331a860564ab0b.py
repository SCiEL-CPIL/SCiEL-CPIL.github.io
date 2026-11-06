"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import bisect

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # 1. Sieve of Eratosthenes for primes up to 10000
    LIMIT = 10000
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, math.isqrt(LIMIT) + 1):
        if is_prime[i]:
            for j in range(i * i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, LIMIT + 1) if is_prime[p]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # 2. Single DP pass with parent tracking
    # dp[i] = list of (gap, length, prev_idx), sorted by gap ascending, length strictly increasing
    # We only run the DP once and track parents for reconstruction.
    
    dp = [[] for _ in range(N)]  # dp[i] = list of (gap, length, prev_idx)
    best_len = 1
    best_idx = 0
    
    for i in range(N):
        if time.time() > deadline - 3.0:
            break
        
        candidates = [(0, 1, -1)]  # (last_gap, length, prev_prime_index)
        
        for j in range(i):
            new_gap = primes[i] - primes[j]
            # For dp[j], entries are sorted by gap ascending.
            # We need entries where prev_gap <= new_gap.
            # Since gaps are sorted ascending, we can binary search for the cutoff.
            # But the frontier is small (typically < 20 entries), so linear scan is fine.
            # Still, let's use bisect for the gap component.
            dj = dp[j]
            if not dj:
                continue
            # Find how many entries have gap <= new_gap
            # dj is sorted by gap ascending
            # Use bisect_right on the gap values
            # Extract gaps for binary search
            # Since dj is small, just scan
            for k in range(len(dj)):
                prev_gap = dj[k][0]
                if prev_gap > new_gap:
                    break  # no more entries will satisfy
                prev_len = dj[k][1]
                candidates.append((new_gap, prev_len + 1, j))
        
        # Compute Pareto frontier: sort by gap ascending, then by length descending
        # Keep only entries where length is strictly increasing (as gap increases)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        frontier = []
        max_len_so_far = -1
        for gap, length, prev_idx in candidates:
            if length > max_len_so_far:
                frontier.append((gap, length, prev_idx))
                max_len_so_far = length
        
        dp[i] = frontier
        
        # Track best
        if frontier:
            best_here = frontier[-1][1]  # largest length in frontier
            if best_here > best_len:
                best_len = best_here
                best_idx = i
    
    # Reconstruct sequence from best_idx
    if best_len <= 1:
        return [primes[best_idx]]
    
    seq = [primes[best_idx]]
    cur = best_idx
    while dp[cur]:
        # The last entry in the frontier has the max length
        gap, length, prev_idx = dp[cur][-1]
        if prev_idx == -1:
            break
        seq.append(primes[prev_idx])
        cur = prev_idx
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
