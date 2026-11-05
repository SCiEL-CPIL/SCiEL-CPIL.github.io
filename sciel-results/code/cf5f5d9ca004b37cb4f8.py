import time
import math
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 10000
    DEADLINE = 50.0  # Safety margin
    
    # Generate primes
    sieve = [True] * (LIMIT + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if sieve[i]:
            for j in range(i*i, LIMIT + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    N = len(primes)
    
    # S[i] will be a list of (gap, max_len, best_j)
    # Sorted by gap. max_len is prefix max.
    S = [None] * N
    dp_len = [0] * N
    
    for i in range(N):
        if time.time() - start_time > DEADLINE:
            break
            
        p_i = primes[i]
        raw_candidates = []
        # Base case: length 1, gap 0 (conceptually -inf), j = -1
        raw_candidates.append((0, 1, -1))
        
        max_len_i = 1
        
        for j in range(i):
            g = p_i - primes[j]
            # Query S[j] for max len with gap <= g
            # S[j] is list of (gap, max_len, best_j)
            # Binary search for rightmost gap <= g
            Sj = S[j]
            if Sj is None:
                continue
                
            # Find index of rightmost element with gap <= g
            # Gaps are sorted
            lo, hi = 0, len(Sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if Sj[mid][0] <= g:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo - 1
            if idx >= 0:
                prev_len = Sj[idx][1]
                new_len = prev_len + 1
                if new_len > max_len_i:
                    max_len_i = new_len
                raw_candidates.append((g, new_len, j))
        
        # Sort raw_candidates by gap
        raw_candidates.sort(key=lambda x: x[0])
        
        # Build S[i] with prefix max
        S_i = []
        current_max = 0
        current_best_j = -1
        
        for g, l, j in raw_candidates:
            if l > current_max:
                current_max = l
                current_best_j = j
            # We store the state. 
            # Note: We can compress if multiple entries have same gap?
            # The sort handles it. If same gap, the last one in sort order (stable?) 
            # actually if same gap, we just want the max len.
            # The loop above keeps the max.
            # But if we have (5, 10, j1) and (5, 12, j2), sorted by gap.
            # First (5, 10, j1): current_max=10, best_j=j1. Append (5, 10, j1).
            # Second (5, 12, j2): current_max=12, best_j=j2. Append (5, 12, j2).
            # This is fine. Query will find the last one for gap 5?
            # Binary search finds rightmost index with gap <= g.
            # So it will find the second (5, 12, j2). Correct.
            
            S_i.append((g, current_max, current_best_j))
        
        S[i] = S_i
        dp_len[i] = current_max
        
    # Find global max
    max_total_len = 0
    end_idx = -1
    for i in range(N):
        if dp_len[i] > max_total_len:
            max_total_len = dp_len[i]
            end_idx = i
            
    if end_idx == -1:
        return []
        
    # Reconstruct path
    path = []
    curr_i = end_idx
    while curr_i != -1:
        path.append(primes[curr_i])
        # Get best_j from the last element of S[curr_i]
        # The last element has the max length for this prime
        if S[curr_i]:
            curr_i = S[curr_i][-1][2]
        else:
            break
            
    path.reverse()
    return path

if __name__ == "__main__":
    res = construct()
    print(res)
