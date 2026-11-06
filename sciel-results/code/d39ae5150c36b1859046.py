import time
from math import isqrt
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N_LIMIT = 10000
    
    # 1. Sieve of Eratosthenes
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N_LIMIT) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n < 2:
        return primes
    
    # DP[i][j] = max length of valid sequence ending with primes[j], primes[i]
    # j < i
    # We only need j < i
    # Initialize DP as a list of lists of zeros
    # To save memory and time, we can use a flat list or just list of lists
    # n is ~1229, so n*n is ~1.5M, which is fine.
    
    DP = [[0] * n for _ in range(n)]
    
    # SuffixMax[j] will be a list of length j
    # SuffixMax[j][k] = max(DP[j][t] for t in range(k, j))
    # We compute SuffixMax[j] after DP[j] is fully computed.
    # DP[j] is computed when outer loop i = j.
    # So when we are at outer loop i, SuffixMax[0] ... SuffixMax[i-1] are ready.
    
    SuffixMax = [None] * n
    
    best_len = 2
    best_i = 1
    best_j = 0
    
    # Precompute gaps? No, primes[i] - primes[j] is fast enough.
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        # Compute SuffixMax for i-1 if i > 1
        # Actually, we need SuffixMax[j] for all j < i.
        # SuffixMax[i-1] becomes available after row DP[i-1] is computed.
        # Row DP[i-1] was computed in the previous iteration (when i was i-1).
        # So we can compute SuffixMax[i-1] at the beginning of iteration i.
        
        if i > 1:
            # Compute SuffixMax[i-1]
            # DP[i-1] has valid entries for k in 0 .. i-2
            # We need SuffixMax[i-1][k] for k in 0 .. i-2
            sm = [0] * (i - 1)
            max_val = 0
            # Iterate k from i-2 down to 0
            for k in range(i - 2, -1, -1):
                if DP[i-1][k] > max_val:
                    max_val = DP[i-1][k]
                sm[k] = max_val
            SuffixMax[i - 1] = sm
        
        p_i = primes[i]
        
        for j in range(i):
            if time.time() > deadline:
                break
                
            p_j = primes[j]
            gap_ij = p_i - p_j
            
            # We need max DP[j][k] for k < j such that p_j - p_k <= gap_ij
            # p_k >= p_j - gap_ij = 2*p_j - p_i
            target = 2 * p_j - p_i
            
            # Find smallest k in [0, j-1] such that primes[k] >= target
            # bisect_left returns the index where target would be inserted to maintain order
            # We search in primes[0:j]
            k_start = bisect_left(primes, target, 0, j)
            
            if k_start < j:
                # Valid predecessors are k in [k_start, j-1]
                # We need max DP[j][k] for k in [k_start, j-1]
                # This is exactly SuffixMax[j][k_start]
                # Note: SuffixMax[j] is computed when outer loop was j+1?
                # No, SuffixMax[j] is computed when outer loop index is j+1.
                # Since we are at i, and j < i, we have i >= j+1.
                # So SuffixMax[j] is already computed.
                max_prev_len = SuffixMax[j][k_start]
                DP[i][j] = 1 + max_prev_len
            else:
                # No valid predecessor k < j
                DP[i][j] = 2
                
            if DP[i][j] > best_len:
                best_len = DP[i][j]
                best_i = i
                best_j = j

    # Reconstruct sequence
    # We end at (best_i, best_j)
    # The sequence ends with primes[best_j], primes[best_i]
    
    path = [best_i, best_j]
    curr_i = best_i
    curr_j = best_j
    
    while len(path) < best_len:
        # We need to find the k that led to DP[curr_i][curr_j]
        # DP[curr_i][curr_j] = 1 + max(DP[curr_j][k] for k in [k_start, curr_j-1])
        # Or 2 if no k.
        # If DP[curr_i][curr_j] == 2, we are done (length 2 means 2 elements, path has 2)
        if DP[curr_i][curr_j] == 2:
            break
            
        # It must be > 2, so there is a predecessor
        p_i = primes[curr_i]
        p_j = primes[curr_j]
        gap_ij = p_i - p_j
        target = 2 * p_j - p_i
        k_start = bisect_left(primes, target, 0, curr_j)
        
        # Find any k in [k_start, curr_j-1] such that DP[curr_j][k] == DP[curr_i][curr_j] - 1
        target_len = DP[curr_i][curr_j] - 1
        found_k = -1
        for k in range(k_start, curr_j):
            if DP[curr_j][k] == target_len:
                found_k = k
                break
                
        if found_k == -1:
            # Should not happen, but break to avoid infinite loop
            break
            
        path.append(found_k)
        curr_i = curr_j
        curr_j = found_k
        
    # Reverse path to get increasing order
    path.reverse()
    
    return [primes[idx] for idx in path]
