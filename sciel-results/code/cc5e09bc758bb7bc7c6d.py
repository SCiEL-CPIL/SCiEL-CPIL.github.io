import bisect
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 90.0  # 90s limit, aim for 85s
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    # If fewer than 2 primes, return them
    if n < 2:
        return primes
        
    # dp[i][j] stores length of sequence ending at primes[i], primes[j]
    # We store dp as list of lists. dp[j] is list of length j, containing dp[0][j], ..., dp[j-1][j]
    # parent[i][j] stores the index k such that dp[i][j] = 1 + dp[k][i]
    # parent[j] is list of length j, containing parent[0][j], ..., parent[j-1][j]
    
    dp = [None] * n
    parent = [None] * n
    
    # dp[j] will be filled when we process i < j
    # Actually, we iterate i from 0 to n-1.
    # For each i, we have dp[k][i] for all k < i already computed.
    # We use these to compute dp[i][j] for all j > i.
    
    max_len = 2
    max_i = 0
    max_j = 1
    
    for i in range(n):
        # Prepare suffix max arrays for dp[k][i] where k < i
        # If i == 0, there are no k < i.
        if i > 0:
            # A[k] = dp[k][i]
            # dp[i] is not computed yet, but dp[0..i-1] are lists containing values for index i?
            # Wait, my storage scheme: dp[j] contains dp[k][j] for k<j.
            # So to get dp[k][i], I look at dp[i][k]? No.
            # Let's clarify storage.
            # Let's use a 2D list `dp_table` where `dp_table[i][j]` is the length for pair (i,j).
            # But n=1229, n^2 = 1.5M. A list of lists of ints is fine.
            
            pass # Re-evaluating storage strategy for clarity
            
    # Let's restart the DP structure definition for clarity in the code below.
    # dp[i][j] for i < j.
    # We will iterate i from 0 to n-1.
    # For a fixed i, we need to access dp[k][i] for k < i.
    # These are stored in dp[k][i].
    # So we need a way to quickly get max dp[k][i] for k in a range.
    
    # Let's stick to the previous plan:
    # dp is a list of lists. dp[j] is a list of length j.
    # dp[j][k] = length of sequence ending at primes[k], primes[j]. (Note: k < j)
    # So dp[j] has indices 0 to j-1.
    # parent[j][k] = index m < k such that dp[j][k] = 1 + dp[k][m].
    
    dp = [None] * n
    parent = [None] * n
    
    # Initialize for j=1 (i=0)
    # dp[1] has length 1. dp[1][0] = 2.
    if n > 1:
        dp[1] = [2]
        parent[1] = [-1] # No parent, just start
        
        max_len = 2
        max_i = 0
        max_j = 1

    for i in range(1, n):
        # We are computing dp[j][i] for all j > i?
        # No, the recurrence is dp[i][j] depends on dp[k][i].
        # So if we fix i, we compute dp[i][j] for all j > i.
        # But dp[i][j] is stored in dp[j][i].
        
        # So for a fixed i, we look at all k < i.
        # We have dp[k][i] stored in dp[i][k].
        # We build suffix max structures on the array V where V[k] = dp[i][k] for k=0..i-1.
        
        if i == 0:
            # No k < 0.
            # For all j > 0, dp[j][0] = 2.
            for j in range(1, n):
                dp[j] = [0] * j
                parent[j] = [-1] * j
                dp[j][0] = 2
                # parent is -1
        else:
            # Build V = [dp[i][0], dp[i][1], ..., dp[i][i-1]]
            # Note: dp[i] is a list of length i.
            V = dp[i]
            
            # Compute suffix max values and indices
            # val_suffix[m] = max(V[m], ..., V[i-1])
            # idx_suffix[m] = argmax index
            val_suffix = [0] * i
            idx_suffix = [0] * i
            
            val_suffix[i-1] = V[i-1]
            idx_suffix[i-1] = i-1
            
            for m in range(i-2, -1, -1):
                if V[m] > val_suffix[m+1]:
                    val_suffix[m] = V[m]
                    idx_suffix[m] = m
                else:
                    val_suffix[m] = val_suffix[m+1]
                    idx_suffix[m] = idx_suffix[m+1]
            
            # Now compute dp[j][i] for all j > i
            for j in range(i+1, n):
                gap = primes[j] - primes[i]
                # We need P[k] >= P[i] - gap
                L = primes[i] - gap
                
                # Find smallest m in [0, i-1] such that primes[m] >= L
                # bisect_left on primes[0:i]
                m = bisect.bisect_left(primes, L, 0, i)
                
                if m < i:
                    k = idx_suffix[m]
                    length = 1 + val_suffix[m]
                    dp[j] = [0] * j
                    parent[j] = [-1] * j
                    dp[j][i] = length
                    parent[j][i] = k
                    
                    if length > max_len:
                        max_len = length
                        max_i = i
                        max_j = j
                else:
                    # No valid k, sequence is just i, j
                    dp[j] = [0] * j
                    parent[j] = [-1] * j
                    dp[j][i] = 2
                    parent[j][i] = -1
                    
                    if 2 > max_len:
                        max_len = 2
                        max_i = i
                        max_j = j
                        
        # Time check
        if time.time() > deadline - 5.0:
            break

    # If we broke early or didn't update max_len correctly in the first few steps, ensure we have a valid result
    # Actually, if n>=2, max_len is at least 2.
    
    # Reconstruct sequence
    if max_len < 2:
        return primes[:max_len] if max_len > 0 else []
        
    seq = [primes[max_j], primes[max_i]]
    curr_j = max_j
    curr_i = max_i
    
    while parent[curr_j][curr_i] != -1:
        prev_k = parent[curr_j][curr_i]
        seq.append(primes[prev_k])
        curr_j = curr_i
        curr_i = prev_k
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
