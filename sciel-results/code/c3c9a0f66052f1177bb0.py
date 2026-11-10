import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to consider n in [1000, 1000000]
    # So n+i for i=1..24 means we need numbers in [1001, 1000024]
    max_val = 1000024
    min_val = 1001
    
    # Compute small-prime factor for each number in [min_val, max_val]
    # small_part[m] = product of p^v_p(m) for p in small_primes
    
    # Use array indexed by m - min_val
    size = max_val - min_val + 1
    small_part = np.ones(size, dtype=np.int64)
    
    # For each small prime, compute its contribution
    for p in small_primes:
        # For each multiple of p, divide out p and multiply small_part
        # Actually, let's compute v_p(m) for each m
        # More efficient: for each power p^k, mark multiples
        power = p
        while power <= max_val:
            # For each multiple of power, the small_part gets an additional factor of p
            # But we need to be careful: we want p^v_p(m), so for each m, 
            # v_p(m) = sum of 1 for each k such that p^k divides m
            # So we can: for each k, for each multiple of p^k, multiply small_part by p
            for m in range(power, max_val + 1, power):
                if m >= min_val:
                    small_part[m - min_val] *= p
            power *= p
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Now find the sliding window of size 24 with minimum distinct small_part values
    # Window for n: numbers n+1, n+2, ..., n+24
    # n ranges from 1000 to 1000000
    # n+1 ranges from 1001 to 1000001, n+24 ranges from 10024 to 1000024
    
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window with frequency tracking
    from collections import defaultdict
    
    # Initial window: n=1000, numbers 1001 to 1024
    freq = defaultdict(int)
    for i in range(1, 25):
        val = small_part[1000 + i - min_val]
        freq[val] += 1
    
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        # Remove n (which was n-1+1 = n)
        # Window was [n-1+1, n-1+24] = [n, n+23]
        # New window: [n+1, n+24]
        # Remove element at position n (index n - min_val)
        # Add element at position n+24 (index n+24 - min_val)
        
        old_val = small_part[n - min_val]  # this is n+1-1 = n... wait
        # Let me re-index. For window starting at n, elements are n+1, n+2, ..., n+24
        # Index in small_part: (n+i) - min_val for i=1..24
        
        # When sliding from n-1 to n:
        # Remove (n-1)+1 = n, add (n-1)+24 = n+23... no
        # Window at n-1: (n-1)+1 to (n-1)+24 = n to n+23
        # Window at n: n+1 to n+24
        # So remove n, add n+24
        
        if n - min_val >= 0 and n - min_val < size:
            old_val = small_part[n - min_val]
            freq[old_val] -= 1
            if freq[old_val] == 0:
                del freq[old_val]
        
        if n + 24 - min_val >= 0 and n + 24 - min_val < size:
            new_val = small_part[n + 24 - min_val]
            freq[new_val] += 1
        
        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
        
        if time.time() > deadline - 2:
            break
    
    return best_n
