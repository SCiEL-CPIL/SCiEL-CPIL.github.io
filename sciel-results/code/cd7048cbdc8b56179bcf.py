"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 1000000
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute the small-prime part for each number in [1, N+24]
    # small_part[m] = m with all prime factors >= 24 removed
    # i.e., the largest divisor of m composed only of primes < 24
    size = N + 25  # we need up to N + 24
    small_part = np.ones(size, dtype=np.int64)
    
    # Sieve-like approach: for each small prime, divide out all powers
    # Actually, we want the product of p^k for each small prime p dividing m
    # Let's compute it directly: for each m, small_part[m] = product of p^k where p < 24 and p^k || m
    
    # More efficient: initialize small_part[m] = m, then for each large prime p >= 24,
    # divide out all powers of p from small_part[m] for multiples of p.
    # But that's expensive for large primes.
    
    # Alternative: for each m, compute small_part by trial division with small primes only.
    # But doing this for 1M numbers is slow in pure Python.
    
    # Better approach: use the fact that small_part[m] = m / large_part[m]
    # where large_part[m] is the part of m with primes >= 24.
    # 
    # Let's use a modified sieve:
    # 1. Start with small_part[m] = m for all m
    # 2. For each prime p >= 24 up to sqrt(N+24), for each multiple m of p,
    #    divide small_part[m] by p while divisible.
    # 3. For remaining numbers, if small_part[m] > 1 and small_part[m] >= 24 and is prime,
    #    divide it out.
    
    # Actually, simpler: for each m from 2 to size-1, compute small_part[m]
    # by repeatedly dividing by small primes.
    
    # Let's use numpy for speed. Compute small_part for all numbers.
    # For each small prime p, we can compute the p-part of each number.
    
    # Initialize: small_part[m] = 1 for all m
    small_part = np.ones(size, dtype=np.int64)
    
    # For each small prime p, compute p^k part of each number
    for p in small_primes:
        p_power = np.ones(size, dtype=np.int64)
        # For multiples of p, compute the p-part
        for m in range(p, size, p):
            temp = m
            pp = 1
            while temp % p == 0:
                temp //= p
                pp *= p
            p_power[m] = pp
        small_part = small_part * p_power
    
    # Now small_part[m] is the complete small-prime factor of m
    # We need windows of 24 consecutive numbers: n+1, ..., n+24
    # For n in [1000, 1000000], the window is small_part[n+1:n+25]
    
    # Use sliding window with frequency count
    # Window size = 24
    window_size = 24
    
    # We'll scan n from 1000 to 1000000
    # For each n, the window is indices n+1 to n+24 (inclusive)
    
    # Use a dict to track frequency of values in current window
    from collections import defaultdict
    
    best_n = 1000
    best_distinct = float('inf')
    
    # Initialize window for n = 1000: indices 1001 to 1024
    freq = defaultdict(int)
    for i in range(1001, 1001 + window_size):
        val = small_part[i]
        freq[val] += 1
    
    distinct_count = len(freq)
    if distinct_count < best_distinct:
        best_distinct = distinct_count
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        # Remove small_part[n] (the element leaving)
        old_val = small_part[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        # Add small_part[n + window_size] (the element entering)
        new_val = small_part[n + window_size]
        freq[new_val] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_n

# EVOLVE-BLOCK-END
