import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute the small-prime factor for each number in [1001, 1000024]
    # n is in [1000, 1000000], n+i in [1001, 1000024]
    max_val = 1000024
    min_val = 1001
    
    # small_factor[m] = product of p^k for each prime p < 24 where p^k divides m
    small_factor = [1] * (max_val + 1)
    
    for p in primes:
        for m in range(p, max_val + 1, p):
            # Divide out all powers of p
            temp = m
            while temp % p == 0:
                small_factor[m] *= p
                temp //= p
    
    # Now slide window: for each n in [1000, 1000000],
    # the 24 numbers are n+1, n+2, ..., n+24
    # Extract small_factor[n+1], ..., small_factor[n+24]
    # Count distinct values
    
    best_n = 1000
    best_count = 25  # worst case is 24 distinct, so start at 25
    
    # Use a frequency counter approach for sliding window
    freq = Counter()
    
    # Initialize window for n=1000: numbers 1001..1024
    for i in range(1, 25):
        val = small_factor[1000 + i]
        freq[val] += 1
    
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        # Remove n (which is n-1+1 = previous window's first element)
        # Previous window was (n-1)+1 .. (n-1)+24 = n .. n+23
        # New window is n+1 .. n+24
        # So remove small_factor[n], add small_factor[n+24]
        
        old_val = small_factor[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        new_val = small_factor[n + 24]
        freq[new_val] += 1
        
        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break
        
        # Time check
        if time.time() > deadline:
            break
    
    return best_n
