"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    def primes_upto(n):
        """Sieve of Eratosthenes."""
        if n < 2:
            return []
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0] = 0
        if n >= 1:
            sieve[1] = 0
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n + 1) if sieve[i]]
    
    # Generate primes up to 10000
    primes = primes_upto(10000)
    m = len(primes)
    
    if m < 2:
        return primes
    
    # DP with Pareto frontier and parent tracking
    # dp[j] = sorted list of (gap, length) Pareto-optimal states ending at primes[j]
    # parent[j][gap] = (prev_index, prev_gap) for reconstruction
    dp = [[] for _ in range(m)]
    parent = [{} for _ in range(m)]
    
    # Base case: single element sequences (gap=0, length=1)
    dp[0] = [(0, 1)]
    parent[0][0] = (-1, 0)
    
    best_len = 1
    best_end = 0
    best_gap = 0
    
    bisect_right = bisect.bisect_right
    inf = float('inf')
    
    for j in range(1, m):
        candidates = []
        p_j = primes[j]
        
        for i in range(j):
            d = p_j - primes[i]
            # Find max length in dp[i] with last_gap <= d
            lst = dp[i]
            if not lst:
                continue
            
            # Binary search for rightmost entry with gap <= d
            idx = bisect_right(lst, (d, inf)) - 1
            if idx >= 0:
                prev_gap, prev_len = lst[idx]
                new_len = prev_len + 1
                candidates.append((d, new_len, i, prev_gap))
        
        if not candidates:
            dp[j] = [(0, 1)]
            parent[j][0] = (-1, 0)
            continue
        
        # Build Pareto frontier: sort by gap, keep only strictly increasing length
        candidates.sort(key=lambda x: (x[0], -x[1]))
        frontier = []
        max_len = -1
        par_j = parent[j]
        for gap, length, prev_i, prev_gap in candidates:
            if length > max_len:
                frontier.append((gap, length))
                par_j[gap] = (prev_i, prev_gap)
                max_len = length
        
        dp[j] = frontier
        
        # Update best
        if frontier:
            last_gap, last_len = frontier[-1]
            if last_len > best_len:
                best_len = last_len
                best_end = j
                best_gap = last_gap
        
        # Deadline check
        if time.time() - t0 > DEADLINE:
            break
    
    # Reconstruct the sequence
    seq = [primes[best_end]]
    cur_idx = best_end
    cur_gap = best_gap
    
    while cur_idx != -1 and cur_gap in parent[cur_idx]:
        prev_i, prev_gap = parent[cur_idx][cur_gap]
        if prev_i == -1:
            break
        seq.append(primes[prev_i])
        cur_idx = prev_i
        cur_gap = prev_gap
    
    seq.reverse()
    
    # Verify the sequence has nondecreasing gaps
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                # Fallback to greedy
                values = [primes[0]]
                for p in primes[1:]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                return values
    
    return seq

# EVOLVE-BLOCK-END
