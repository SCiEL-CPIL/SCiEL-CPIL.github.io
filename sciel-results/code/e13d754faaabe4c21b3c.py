"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to evaluate n in [1000, 1000000]
    # For each n, we look at n+1, n+2, ..., n+24
    # So we need small-prime parts of numbers in [1001, 1000024]
    
    max_val = 1000000 + 24  # 1000024
    min_val = 1000 + 1      # 1001
    
    # Precompute small-prime part for all numbers in [1, max_val]
    # small_part[m] = largest divisor of m composed only of small_primes
    small_part = list(range(max_val + 1))
    
    # Sieve: for each small prime, divide it out
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            # Divide out all factors of p from small_part[multiple]
            # Actually, we want the LARGEST divisor composed of small primes
            # So small_part[m] should be m with all large-prime factors removed
            # Better approach: start with m, remove large prime factors
            pass
    
    # Better approach: for each number, compute its small-prime part
    # small_part[m] = m / (largest divisor of m coprime to product of small primes)
    # Equivalently: remove all prime factors >= 24 from m
    
    # Let's use a different sieve approach:
    # Initialize small_part[m] = m for all m
    # For each prime p >= 24 (i.e., p not in small_primes), divide p out of all multiples
    # But we don't want to iterate over all primes >= 24 up to max_val
    
    # Alternative: for each m, factor out small primes to get the "large part"
    # small_part[m] = m / large_part[m]
    
    # Most efficient: use a sieve to compute the "rough part" (part with primes >= 24)
    # Then small_part[m] = m // rough_part[m]
    
    # Let's precompute using a modified sieve
    # rough_part[m] = largest divisor of m whose prime factors are all >= 24
    # small_part[m] = m // rough_part[m]
    
    # Initialize rough_part to 1
    rough_part = [1] * (max_val + 1)
    
    # For each prime p >= 24, multiply p into rough_part of its multiples
    # We need primes from 29 to max_val (29 is the first prime >= 24)
    # Actually primes >= 24: 29, 31, 37, ... (24,25,26,27,28 are not prime; 29 is first)
    
    # Generate primes from 29 to max_val using simple sieve
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    # For each prime p >= 29, update rough_part for multiples of p
    for p in range(29, max_val + 1):
        if is_prime[p]:
            for multiple in range(p, max_val + 1, p):
                rough_part[multiple] *= p
    
    # Now small_part[m] = m // rough_part[m]
    # Compute this for the range we need
    # Actually let's just compute it on the fly for the window
    
    # Now search for the best n
    best_n = 1000
    best_count = float('inf')
    
    # For efficiency, precompute small_part for the needed range
    # We need small_part for values in [1001, 1000024]
    needed_min = min_val
    needed_max = max_val
    
    small_part_arr = [0] * (needed_max + 1)
    for m in range(needed_min, needed_max + 1):
        small_part_arr[m] = m // rough_part[m]
    
    # Now slide window of size 24
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        # Count distinct small_part values for n+1, ..., n+24
        vals = set()
        for i in range(1, 25):
            vals.add(small_part_arr[n + i])
        
        cnt = len(vals)
        if cnt < best_count:
            best_count = cnt
            best_n = n
            if best_count <= 1:
                break  # Can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
