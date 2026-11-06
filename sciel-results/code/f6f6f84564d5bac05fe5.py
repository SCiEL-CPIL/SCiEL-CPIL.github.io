import time
import math

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N <= 1:
        return primes
    
    # dp[i][j] = max length of valid subsequence ending at primes[i], primes[j] (i < j)
    # We store dp as a 2D list, dp[i][j] for i < j
    dp = [[0] * N for _ in range(N)]
    
    # For each i, suffix_max[i][k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
    # suffix_max[i] is an array of length i, where suffix_max[i][k] = max(dp[k][i] for k >= k)
    
    # Initialize: all pairs have length 2
    # Process in order of increasing gap or by j
    # For j from 1 to N-1:
    #   For i from 0 to j-1:
    #     threshold = 2*primes[i] - primes[j]
    #     Find k_min = smallest k such that primes[k] >= threshold
    #     If k_min < i: dp[i][j] = 1 + suffix_max[i][k_min]
    #     Else: dp[i][j] = 2
    #     But we need suffix_max[i] to be computed before using it for j
    
    # Actually, for dp[i][j], we need dp[k][i] for k < i. These are computed when we processed index i as the "j" in earlier iterations.
    # So we process j from 1 to N-1, and for each j, process i from 0 to j-1.
    # Before processing any i for a given j, we need suffix_max for all i < j to be ready.
    # suffix_max[i] depends on dp[k][i] for k < i, which are computed when j=i was processed.
    
    # Let's restructure:
    # After computing all dp[k][i] for k < i (i.e., when we finished processing j=i), compute suffix_max[i].
    
    # Initialize dp for length-2 sequences
    for i in range(N):
        for j in range(i + 1, N):
            dp[i][j] = 2
    
    # suffix_max[i] = array where suffix_max[i][k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
    # We'll compute suffix_max[i] after all dp[k][i] are set (k < i)
    
    # But dp[i][j] for i < j needs suffix_max[i], which requires dp[k][i] for k < i to be finalized.
    # dp[k][i] is finalized when we process j=i.
    # So: process j from 1 to N-1. For each j, first compute suffix_max for all i < j that haven't been computed yet? 
    # Actually, suffix_max[i] only depends on dp[k][i] for k < i, which are all set when j=i was processed.
    # So after processing j=i (setting dp[k][i] for all k < i), we can compute suffix_max[i].
    
    # Revised plan:
    # dp[i][j] = 2 initially for all i < j
    # For j from 1 to N-1:
    #   # Compute suffix_max for index j-1? No, suffix_max[i] for i < j
    #   Actually, for computing dp[i][j], we need suffix_max[i] which uses dp[k][i] for k < i.
    #   dp[k][i] were computed in previous iterations (when j was i).
    #   So before the inner loop over i for a given j, compute suffix_max for all i in [0, j-1] if not done.
    #   But suffix_max[i] only changes when dp[k][i] changes, which happens when j=i is processed.
    #   So compute suffix_max[i] right after finishing processing j=i.
    
    # Let me just compute suffix_max[i] on the fly.
    
    # For each i, suffix_max[i][k] for k in range(i)
    # suffix_max[i] = [0] * i
    # suffix_max[i][i-1] = dp[i-1][i]
    # suffix_max[i][k] = max(dp[k][i], suffix_max[i][k+1])
    
    # We'll store suffix_max as a list of lists.
    
    suffix_max = [None] * N  # suffix_max[i] will be a list of length i
    
    # Process j from 1 to N-1
    best_len = 2
    best_end = (0, 1)  # indices (i, j)
    
    for j in range(1, N):
        # Compute suffix_max for all i < j that we haven't computed yet
        # Actually, suffix_max[i] is needed for dp[i][j], and suffix_max[i] depends on dp[k][i] for k < i
        # dp[k][i] are finalized after processing j=i
        # So for current j, all i < j have their dp[k][i] finalized (since i < j, j=i was processed earlier)
        # So we can compute suffix_max[i] for i in range(j) if not computed
        
        for i in range(j):
            if suffix_max[i] is None:
                # Compute suffix_max[i]
                sm = [0] * i
                if i > 0:
                    sm[i-1] = dp[i-1][i]
                    for k in range(i-2, -1, -1):
                        sm[k] = max(dp[k][i], sm[k+1])
                suffix_max[i] = sm
        
        # Now compute dp[i][j] for i in range(j)
        for i in range(j):
            if time.time() > deadline:
                break
            threshold = 2 * primes[i] - primes[j]
            # Find smallest k such that primes[k] >= threshold
            # Binary search in primes[0:i] for threshold
            lo, hi = 0, i  # search in [0, i)
            k_min = i  # if not found, k_min = i means no valid k
            while lo < hi:
                mid = (lo + hi) // 2
                if primes[mid] >= threshold:
                    hi = mid
                else:
                    lo = mid + 1
            k_min = lo
            
            if k_min < i:
                # dp[i][j] = 1 + suffix_max[i][k_min]
                val = 1 + suffix_max[i][k_min]
                if val > dp[i][j]:
                    dp[i][j] = val
            # else dp[i][j] stays 2
            
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_end = (i, j)
        if time.time() > deadline:
            break
    
    # Reconstruct the sequence
    # We have best_end = (i, j) with dp[i][j] = best_len
    # We need to backtrack
    if best_len <= 1:
        return primes[:best_len]
    
    # Backtrack: find the sequence
    # dp[i][j] = 1 + dp[k][i] for some k, or 2
    # We need to store the parent pointer
    
    # Let's redo with parent pointers
    # Actually, let's just reconstruct from dp
    
    # For the best pair (i, j):
    # If dp[i][j] == 2: sequence is [primes[i], primes[j]]
    # Else: dp[i][j] = 1 + dp[k][i] for some k with primes[k] >= 2*primes[i] - primes[j]
    #   We need to find which k achieved the max
    
    # Let's add parent tracking
    # parent[i][j] = the k that gave the max dp[k][i] for dp[i][j]
    
    # Redo with parent tracking
    dp2 = [[2] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]  # parent[i][j] = k
    suffix_max2 = [None] * N
    suffix_max_parent = [None] * N  # suffix_max_parent[i][k] = the argmax index
    
    best_len = 2
    best_end = (0, 1)
    
    for j in range(1, N):
        for i in range(j):
            if time.time() > deadline:
                break
            if suffix_max2[i] is None:
                sm = [0] * i
                sm_parent = [-1] * i
                if i > 0:
                    sm[i-1] = dp2[i-1][i]
                    sm_parent[i-1] = i-1
                    for k in range(i-2, -1, -1):
                        if dp2[k][i] > sm[k+1]:
                            sm[k] = dp2[k][i]
                            sm_parent[k] = k
                        else:
                            sm[k] = sm[k+1]
                            sm_parent[k] = sm_parent[k+1]
                suffix_max2[i] = sm
                suffix_max_parent[i] = sm_parent
        
        for i in range(j):
            if time.time() > deadline:
                break
            threshold = 2 * primes[i] - primes[j]
            lo, hi = 0, i
            while lo < hi:
                mid = (lo + hi) // 2
                if primes[mid] >= threshold:
                    hi = mid
                else:
                    lo = mid + 1
            k_min = lo
            
            if k_min < i:
                val = 1 + suffix_max2[i][k_min]
                if val > dp2[i][j]:
                    dp2[i][j] = val
                    parent[i][j] = suffix_max_parent[i][k_min]
            
            if dp2[i][j] > best_len:
                best_len = dp2[i][j]
                best_end = (i, j)
        if time.time() > deadline:
            break
    
    # Reconstruct
    seq = []
    i, j = best_end
    seq.append(primes[j])
    seq.append(primes[i])
    while parent[i][j] != -1:
        k = parent[i][j]
        seq.append(primes[k])
        j = i
        i = k
        # Now we need parent[i][j] for the new pair
        # Wait, this is getting complicated. Let me rethink.
    
    # Actually, let me just return the greedy result if DP is too complex, or simplify
    # Given time constraints, let me just use the DP without backtracking and find the max length,
    # then do a separate reconstruction.
    
    # For now, let me just return based on best_len found
    # Actually, I need to return the actual sequence. Let me do proper backtracking.
    
    # Reconstruct properly
    result = []
    ci, cj = best_end
    result.append(primes[cj])
    result.append(primes[ci])
    
    while parent[ci][cj] != -1:
        ck = parent[ci][cj]
        result.append(primes[ck])
        # The previous pair was (ck, ci)
        cj = ci
        ci = ck
    
    result.reverse()
    return result
# EVOLVE-BLOCK-END
