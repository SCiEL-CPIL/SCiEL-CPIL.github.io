import bisect
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p*p, limit + 1, p):
                is_prime[multiple] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    # DP state
    # best_ends[i] is a list of (gap, max_len) pairs, sorted by gap.
    # It represents the function M_i(d) = max length of valid sequence ending at i with last gap <= d.
    # Only "record" breaking points are stored.
    best_ends = [[] for _ in range(n)]
    max_lengths = [1] * n
    backtrack = [None] * n  # (prev_index, gap)
    
    # 2. DP Iteration
    for i in range(n):
        if time.time() > deadline:
            break
            
        candidates = []
        pi = primes[i]
        
        for j in range(i):
            d = pi - primes[j]
            
            # Query best_ends[j] for max length with gap <= d
            # best_ends[j] is sorted list of (gap, len)
            # We want the last element with gap <= d
            # Use bisect_right with a dummy large length to ensure we land after all entries with gap == d
            # Since we want gap <= d, and tuples are (gap, len), 
            # bisect_right(list, (d, 10**9)) gives index of first element > (d, 10**9)
            # Since len is always < 10**9, this effectively finds the position after all entries with gap == d.
            # So idx-1 is the last entry with gap <= d.
            
            if not best_ends[j]:
                l_prev = 1
            else:
                idx = bisect.bisect_right(best_ends[j], (d, 10**9))
                if idx > 0:
                    l_prev = best_ends[j][idx-1][1]
                else:
                    l_prev = 1
            
            l_new = l_prev + 1
            candidates.append((d, l_new, j))
        
        # Process candidates to build best_ends[i]
        if candidates:
            # Sort by gap
            candidates.sort(key=lambda x: x[0])
            
            current_max = 1
            best_j_for_max = None
            best_d_for_max = None
            best_ends_i = []
            
            for d, l, j in candidates:
                if l > current_max:
                    current_max = l
                    best_j_for_max = j
                    best_d_for_max = d
                    best_ends_i.append((d, l))
            
            best_ends[i] = best_ends_i
            max_lengths[i] = current_max
            backtrack[i] = (best_j_for_max, best_d_for_max)
        else:
            # i=0 case, or no valid predecessors
            best_ends[i] = []
            max_lengths[i] = 1
            backtrack[i] = None

    # 3. Find best sequence
    if n == 0:
        return []
        
    max_len = 0
    best_end_idx = 0
    for i in range(n):
        if max_lengths[i] > max_len:
            max_len = max_lengths[i]
            best_end_idx = i
            
    # Reconstruct
    path_indices = [best_end_idx]
    cur = best_end_idx
    while backtrack[cur] is not None:
        prev_idx, gap = backtrack[cur]
        path_indices.append(prev_idx)
        cur = prev_idx
        
    path_indices.reverse()
    sequence = [primes[idx] for idx in path_indices]
    
    return sequence
