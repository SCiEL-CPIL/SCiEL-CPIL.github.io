"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # 1. Generate Primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [primes[0]]
        
    # 2. DP Setup
    # dp[j][i] = length of longest valid subsequence ending with primes[j], primes[i]
    # prev[j][i] = index k such that primes[k] is the element before primes[j] in the optimal subsequence ending with primes[j], primes[i]
    # If prev[j][i] is -1, the subsequence is just [primes[j], primes[i]]
    
    # Initialize dp with 2 (minimum length for a pair)
    dp = [[2] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    max_len = 2
    best_j = 0
    best_i = 1
    
    # Iterate middle element j
    for j in range(N):
        # For each j, we want to compute dp[j][i] for all i > j
        # We need max dp[k][j] for k < j such that primes[k] >= 2*primes[j] - primes[i]
        
        # Pointer k_ptr moves from j-1 downwards
        k_ptr = j - 1
        current_max = 0
        best_k = -1
        
        # We iterate i from j+1 to N-1
        # As i increases, primes[i] increases, so 2*primes[j] - primes[i] decreases.
        # The condition primes[k] >= threshold becomes easier to satisfy.
        # So k_ptr can only move left (decrease).
        
        for i in range(j + 1, N):
            if time.time() > deadline:
                # Return best found so far
                # Reconstructing might take a bit, but it's O(N)
                if max_len == 2:
                    return [primes[best_j], primes[best_i]]
                # If we timeout, we should probably return a valid sequence
                # Let's just break and reconstruct
                break
            
            threshold = 2 * primes[j] - primes[i]
            
            # Add valid k's to current_max
            while k_ptr >= 0 and primes[k_ptr] >= threshold:
                if dp[k_ptr][j] > current_max:
                    current_max = dp[k_ptr][j]
                    best_k = k_ptr
                k_ptr -= 1
            
            if current_max > 0:
                new_len = current_max + 1
                dp[j][i] = new_len
                prev[j][i] = best_k
                if new_len > max_len:
                    max_len = new_len
                    best_j = j
                    best_i = i
            else:
                # No valid predecessor, length is 2
                dp[j][i] = 2
                prev[j][i] = -1
                if 2 > max_len: # This won't happen as max_len starts at 2
                    max_len = 2
                    best_j = j
                    best_i = i

    # 3. Reconstruct Sequence
    # Start from best_i, best_j
    seq = [primes[best_i], primes[best_j]]
    
    j, i = best_j, best_i
    while prev[j][i] != -1:
        k = prev[j][i]
        seq.append(primes[k])
        i = j
        j = k
        
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END
