"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute for each number in range [1001, 1000024] the factor composed of primes < 24
    # Range of n: [1000, 1000000], so n+i ranges from 1001 to 1000024
    min_val = 1001
    max_val = 1000024
    size = max_val - min_val + 1
    
    # factor[i] = the part of (min_val + i) that is composed only of primes < 24
    factors = [1] * size
    
    # For each prime, multiply into all multiples
    for p in primes:
        # For each multiple of p in range, multiply by p repeatedly
        for m in range(min_val, max_val + 1, p):
            idx = m - min_val
            x = m
            while x % p == 0:
                factors[idx] *= p
                x //= p
    
    # Now scan all windows of size 24
    # n ranges from 1000 to 1000000, window covers n+1 to n+24
    # index in factors array for n+i is (n+i) - min_val = n + i - 1001
    
    best_distinct = float('inf')
    best_n = 1000
    
    # Use a frequency dict for the sliding window
    from collections import defaultdict
    freq = defaultdict(int)
    
    # Initialize window for n=1000: covers 1001 to 1024
    # indices 0 to 23
    for i in range(24):
        f = factors[i]
        freq[f] += 1
    
    distinct_count = len(freq)
    if distinct_count < best_distinct:
        best_distinct = distinct_count
        best_n = 1000
    
    # Slide window: n from 1001 to 1000000
    # When n increases by 1, we remove factor at index (n-1+1-1001) = n-1001 and add factor at (n+24-1001) = n+23-1001
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove the element leaving the window: was at n-1, index = (n-1) - 1001 = n - 1002
        remove_idx = n - 1002
        # Add the element entering the window: is at n+24, index = (n+24) - 1001 = n + 23 - 1001 = n - 978
        add_idx = n - 978
        
        f_remove = factors[remove_idx]
        freq[f_remove] -= 1
        if freq[f_remove] == 0:
            del freq[f_remove]
        
        f_add = factors[add_idx]
        freq[f_add] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
