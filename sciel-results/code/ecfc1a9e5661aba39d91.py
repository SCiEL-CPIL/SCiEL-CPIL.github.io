import bisect
import time
import sys

def construct():
    # Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    # dp[i][j] = length of longest sequence ending with primes[i], primes[j]
    # i < j
    dp = [[0] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    # col_suf[i] is a list of length i.
    # col_suf[i][h] stores the index k in [h, i-1] that maximizes dp[k][i].
    # If there are ties, any is fine.
    col_suf = [None] * n
    
    max_len = 2
    best_i = 0
    best_j = 1
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    for j in range(1, n):
        # Check deadline
        if time.time() > deadline:
            break
            
        for i in range(j):
            # Base case is length 2
            best_prev_len = 0
            best_prev_idx = -1
            
            if i > 0:
                # Condition: primes[j] - primes[i] >= primes[i] - primes[h]
                # primes[h] >= 2*primes[i] - primes[j]
                threshold = 2 * primes[i] - primes[j]
                
                # Find smallest h such that primes[h] >= threshold
                # h must be < i
                m = bisect.bisect_left(primes, threshold, 0, i)
                
                if m < i:
                    # We have valid h in [m, i-1]
                    # The best h is stored in col_suf[i][m]
                    h = col_suf[i][m]
                    best_prev_len = dp[h][i]
                    best_prev_idx = h
            
            if best_prev_len > 0:
                dp[i][j] = best_prev_len + 2
                parent[i][j] = best_prev_idx
            else:
                dp[i][j] = 2
                parent[i][j] = -1
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
        
        # Build col_suf[j] for use in future iterations
        # col_suf[j][h] = argmax_{k in [h, j-1]} dp[k][j]
        if j > 0:
            col_suf[j] = [0] * j
            current_best_idx = j - 1
            for h in range(j - 1, -1, -1):
                if h == j - 1:
                    col_suf[j][h] = h
                else:
                    # Compare dp[h][j] and dp[col_suf[j][h+1]][j]
                    idx_next = col_suf[j][h + 1]
                    if dp[h][j] >= dp[idx_next][j]:
                        col_suf[j][h] = h
                    else:
                        col_suf[j][h] = idx_next

    # Reconstruct sequence
    seq = [primes[best_j], primes[best_i]]
    ci, cj = best_i, best_j
    while parent[ci][cj] != -1:
        prev_i = parent[ci][cj]
        seq.append(primes[prev_i])
        ci, cj = prev_i, ci
    
    seq.reverse()
    return seq
