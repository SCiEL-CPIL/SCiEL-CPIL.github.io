import sys
import time
import bisect

def construct():
    start_time = time.time()
    LIMIT = 10000
    
    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    # states[i] is a list of tuples (gap, length, prev_prime_idx, prev_gap)
    # The list is sorted by gap, and length is strictly increasing.
    states = [[] for _ in range(n)]
    
    best_len = 0
    best_end_idx = -1
    best_end_gap = 0
    
    for j in range(n):
        p_j = primes[j]
        cands = []
        
        # Base case: sequence of length 1 starting at p_j
        # Conceptual gap 0
        cands.append((0, 1, -1, 0))
        
        # Extend from previous primes
        for i in range(j):
            p_i = primes[i]
            G = p_j - p_i
            
            # Query states[i] for max length with gap <= G
            # states[i] is sorted by gap
            if not states[i]:
                continue
                
            # Binary search for the largest gap <= G
            # We need the index of the rightmost element with gap <= G
            # bisect_right returns the insertion point. So index = pos - 1
            gaps = [s[0] for s in states[i]]
            pos = bisect.bisect_right(gaps, G)
            if pos > 0:
                best_state = states[i][pos - 1]
                g_old, L_old, prev_idx, prev_gap = best_state
                # New candidate for p_j
                new_len = L_old + 1
                cands.append((G, new_len, i, g_old))
        
        # Consolidate cands
        # Sort by gap
        cands.sort(key=lambda x: x[0])
        
        filtered = []
        max_len_so_far = 0
        
        for c in cands:
            gap, length, prev_idx, prev_gap = c
            if length > max_len_so_far:
                filtered.append((gap, length, prev_idx, prev_gap))
                max_len_so_far = length
        
        states[j] = filtered
        
        if max_len_so_far > best_len:
            best_len = max_len_so_far
            best_end_idx = j
            # We need to know which gap in states[j] corresponds to best_len
            # Since lengths are strictly increasing in filtered, the last one has max length
            # But wait, if multiple cands had the same max length, we kept the one with smallest gap?
            # No, we kept the first one that achieved the new max.
            # Actually, the last element in `filtered` has the max length.
            # Is it possible that an earlier element has the same max length?
            # No, we only append if length > max_len_so_far.
            # So the last element in `filtered` has the unique max length for this j.
            best_end_gap = filtered[-1][0]
            
        # Time check
        if time.time() - start_time > 50:
            break
            
    if best_end_idx == -1:
        return [primes[0]] if primes else []
        
    # Reconstruct sequence
    seq = []
    curr_idx = best_end_idx
    curr_gap = best_end_gap
    
    while curr_idx != -1:
        seq.append(primes[curr_idx])
        
        # Find the state in states[curr_idx] with gap curr_gap
        # Since states[curr_idx] is sorted by gap and gaps are unique, we can binary search
        # Or just linear search since list is small
        state = None
        for s in states[curr_idx]:
            if s[0] == curr_gap:
                state = s
                break
                
        if state is None:
            # Should not happen
            break
            
        prev_idx = state[2]
        prev_gap = state[3]
        
        curr_idx = prev_idx
        curr_gap = prev_gap
        
    seq.reverse()
    
    return seq
