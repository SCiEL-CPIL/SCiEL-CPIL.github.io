"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small_factor using sieve-like approach
    # small_factor[x] = product of all prime factors < 24 (with multiplicity)
    small_factor = [1] * (max_val + 1)
    small_factor[0] = 0
    
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            val = multiple
            while val % p == 0:
                small_factor[multiple] *= p
                val //= p
    
    # Sliding window approach with frequency counting
    # For candidate n, numbers are n+1 .. n+24
    
    best_score = 25
    best_n = 1000
    
    # Initialize window for n=1000: factors of 1001..1024
    freq = Counter()
    for i in range(1, 25):
        freq[small_factor[1000 + i]] += 1
    
    distinct = len(freq)
    if distinct < best_score:
        best_score = distinct
        best_n = 1000
    
    # Slide window: for n from 1001 to 1000000
    # Remove n, add n+24
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove element at position n
        remove_f = small_factor[n]
        freq[remove_f] -= 1
        if freq[remove_f] == 0:
            del freq[remove_f]
        
        # Add element at position n+24
        add_f = small_factor[n + 24]
        freq[add_f] += 1
        
        distinct = len(freq)
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score == 1:
                break
    
    return best_n
# EVOLVE-BLOCK-END
