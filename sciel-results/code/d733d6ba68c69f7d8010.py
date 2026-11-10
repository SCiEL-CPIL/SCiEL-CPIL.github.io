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
    
    # DP arrays
    # dp_gaps[j]: list of gaps (sorted ascending)
    # dp_lens[j]: list of lengths (corresponding to dp_gaps[j], strictly increasing)
    # dp_parents[j]: list of (prev_index, prev_gap) tuples corresponding to dp_gaps[j]
    dp_gaps = [[] for _ in range(m)]
    dp_lens = [[] for _ in range(m)]
    dp_parents = [[] for _ in range(m)]
    
    # Base case: single element sequences
    # Gap 0, Length 1, Parent (-1, 0)
    dp_gaps[0] = [0]
    dp_lens[0] = [1]
    dp_parents[0] = [(-1, 0)]
    
    best_len = 1
    best_end = 0
    best_gap = 0
    
    primes_list = primes
    
    for j in range(1, m):
        pj = primes_list[j]
        candidates = []
        
        # Gather candidates from all i < j
        for i in range(j):
            d = pj - primes_list[i]
            gaps_i = dp_gaps[i]
            if not gaps_i:
                continue
            
            # Find the best state in i with gap <= d
            # bisect_right gives the insertion point for d to maintain sorted order
            # We want the largest gap <= d, so we use bisect_right - 1
            idx = bisect.bisect_right(gaps_i, d) - 1
            if idx >= 0:
                prev_gap = gaps_i[idx]
                prev_len = dp_lens[i][idx]
                new_len = prev_len + 1
                # Store (gap, new_len, prev_index, prev_gap)
                candidates.append((d, new_len, i, prev_gap))
        
        if not candidates:
            # Should not happen for j > 0 if primes are dense enough, but handle anyway
            dp_gaps[j] = [0]
            dp_lens[j] = [1]
            dp_parents[j] = [(-1, 0)]
            continue
        
        # Build Pareto frontier for j
        # Sort by gap, then by length descending (so we pick the longest for a given gap if needed, 
        # but the frontier logic handles this by keeping max length)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        frontier_gaps = []
        frontier_lens = []
        frontier_parents = []
        max_len = -1
        
        for gap, length, prev_i, prev_gap in candidates:
            if length > max_len:
                frontier_gaps.append(gap)
                frontier_lens.append(length)
                frontier_parents.append((prev_i, prev_gap))
                max_len = length
        
        dp_gaps[j] = frontier_gaps
        dp_lens[j] = frontier_lens
        dp_parents[j] = frontier_parents
        
        # Update best
        if frontier_lens:
            last_gap = frontier_gaps[-1]
            last_len = frontier_lens[-1]
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
    
    # Find the index of cur_gap in dp_gaps[cur_idx]
    # Since dp_gaps is sorted, we can use bisect_left to find the exact match
    # Note: There might be multiple states with the same gap? No, frontier keeps unique gaps?
    # Actually, frontier_gaps are unique because we only add if length increases.
    # If two candidates have same gap, the one with higher length is kept.
    # So gaps in frontier are strictly increasing?
    # Wait. If gap A < gap B, len A < len B.
    # Can we have gap A == gap B? 
    # candidates.sort by gap. If gaps are equal, we iterate. 
    # If gap is same, we only add if length > max_len.
    # So for a specific gap value, we only keep the one with the max length.
    # Thus, the list of gaps in the frontier contains unique values?
    # Not necessarily. If we have (gap=5, len=3) and (gap=5, len=4), we keep (5,4).
    # So yes, gaps in the frontier are unique and sorted.
    
    # Use bisect to find the index
    idx_in_frontier = bisect.bisect_left(dp_gaps[cur_idx], cur_gap)
    
    if idx_in_frontier < len(dp_gaps[cur_idx]) and dp_gaps[cur_idx][idx_in_frontier] == cur_gap:
        while True:
            prev_i, prev_gap = dp_parents[cur_idx][idx_in_frontier]
            if prev_i == -1:
                break
            seq.append(primes[prev_i])
            cur_idx = prev_i
            cur_gap = prev_gap
            # Find index in previous frontier
            idx_in_frontier = bisect.bisect_left(dp_gaps[cur_idx], cur_gap)
            if idx_in_frontier >= len(dp_gaps[cur_idx]) or dp_gaps[cur_idx][idx_in_frontier] != cur_gap:
                # Should not happen if data is consistent
                break
    else:
        # Fallback: just return the best prime
        pass

    seq.reverse()
    
    # Verify the sequence has nondecreasing gaps
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                # Fallback to greedy if verification fails (should not happen)
                values = [primes[0]]
                for p in primes[1:]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                return values
    
    return seq

# EVOLVE-BLOCK-END
