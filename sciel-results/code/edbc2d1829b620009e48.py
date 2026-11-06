import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [primes[0]]
        
    # dp[i] will store the length of the longest valid sequence ending at primes[i]
    # prev[i] will store the index of the previous prime in that sequence
    dp = [1] * n
    prev = [-1] * n
    
    # We iterate through primes. For each prime i, we look at all previous primes j
    # to see if we can extend the sequence ending at j to i.
    # Condition: gap(i, j) >= gap(j, prev[j])
    # gap(i, j) = primes[i] - primes[j]
    # gap(j, prev[j]) = primes[j] - primes[prev[j]] if prev[j] != -1 else 0
    
    for i in range(1, n):
        best_len = 1
        best_prev = -1
        p_i = primes[i]
        
        for j in range(i):
            p_j = primes[j]
            gap = p_i - p_j
            
            # Get the previous gap for the sequence ending at j
            if prev[j] != -1:
                prev_gap = p_j - primes[prev[j]]
            else:
                prev_gap = 0
                
            if gap >= prev_gap:
                if dp[j] + 1 > best_len:
                    best_len = dp[j] + 1
                    best_prev = j
                    
        dp[i] = best_len
        prev[i] = best_prev
        
        # Check time
        if time.time() > deadline:
            break
            
    # Find the end of the longest sequence
    max_len = 0
    end_idx = 0
    for i in range(n):
        if dp[i] > max_len:
            max_len = dp[i]
            end_idx = i
            
    # Reconstruct the sequence
    sequence = []
    curr = end_idx
    while curr != -1:
        sequence.append(primes[curr])
        curr = prev[curr]
        
    sequence.reverse()
    return sequence
