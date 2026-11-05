"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    # 2. Dynamic Programming
    # dp[i] = length of the longest valid sequence ending at primes[i]
    # prev[i] = index of the previous prime in that sequence
    dp = [1] * n
    prev = [-1] * n
    
    # We want to maximize the length of the subsequence p_1, ..., p_k
    # such that p_{j+1} - p_j >= p_j - p_{j-1}
    # This is equivalent to: for a sequence ending at j, the previous element i
    # must satisfy: primes[j] - primes[i] >= (primes[i] - primes[prev[i]])
    # Note: If prev[i] is -1, the gap constraint is vacuously true (or 0), 
    # so any i can follow a start.
    
    # Optimization: 
    # For each j, we look for i < j such that:
    # 1. primes[j] - primes[i] >= gap_before_i
    # 2. dp[i] is maximized
    
    # Let gap_before_i be the gap that led to i.
    # If prev[i] == -1, gap_before_i = 0.
    # Else gap_before_i = primes[i] - primes[prev[i]].
    
    # We can iterate j from 0 to n-1.
    # For each j, we check all i < j.
    # O(N^2) where N is number of primes (~1229). 1229^2 ~ 1.5M ops. Very fast.
    
    best_len = 1
    best_end = 0
    
    for j in range(n):
        if time.time() > deadline:
            break
        for i in range(j):
            # Calculate the gap that would be added
            new_gap = primes[j] - primes[i]
            
            # Calculate the gap that existed before i
            if prev[i] == -1:
                old_gap = 0
            else:
                old_gap = primes[i] - primes[prev[i]]
                
            if new_gap >= old_gap:
                if dp[i] + 1 > dp[j]:
                    dp[j] = dp[i] + 1
                    prev[j] = i
                    
        if dp[j] > best_len:
            best_len = dp[j]
            best_end = j
            
    # 3. Reconstruct the sequence
    sequence = []
    curr = best_end
    while curr != -1:
        sequence.append(primes[curr])
        curr = prev[curr]
        
    sequence.reverse()
    return sequence
# EVOLVE-BLOCK-END
