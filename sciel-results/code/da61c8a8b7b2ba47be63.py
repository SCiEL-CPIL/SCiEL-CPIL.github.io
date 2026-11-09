"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_SMALL_PRIME_SET = set(_SMALL_PRIMES)

def _extract_small_factor(m):
    """Extract the part of m composed of primes strictly below 24."""
    result = 1
    for p in _SMALL_PRIMES:
        while m % p == 0:
            result *= p
            m //= p
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute small-prime factors for all numbers in [1001, 1000024]
    # We need factors for n+i where n in [1000, 1000000] and i in [1, 24]
    # So we need factors for numbers in [1001, 1000024]
    
    min_val = 1001
    max_val = 1000024
    size = max_val - min_val + 1
    
    # Precompute array of small-prime factors
    factors = [0] * (size + 1)  # 1-indexed for easier mapping
    for idx in range(size):
        m = min_val + idx
        f = _extract_small_factor(m)
        factors[idx + 1] = f
    
    # Now use sliding window to find minimum distinct count
    # Window is of size 24, covering factors[n-1001+1] to factors[n-1001+24]
    # For n in [1000, 1000000], the window covers factors[n-1000] to factors[n-1000+23]
    # Let's define: for starting index s (0-based in factors array), window covers factors[s+1] to factors[s+24]
    # n = 1000 + s, so s ranges from 0 to 1000000-1000 = 999000
    
    best_n = 1000
    best_val = float('inf')
    
    # Initialize frequency dictionary for first window (s=0, n=1000)
    freq = defaultdict(int)
    distinct_count = 0
    
    # Add first 24 elements
    for i in range(24):
        f = factors[i + 1]
        if freq[f] == 0:
            distinct_count += 1
        freq[f] += 1
    
    if distinct_count < best_val:
        best_val = distinct_count
        best_n = 1000
    
    # Slide the window
    for s in range(1, 999001):  # s from 1 to 999000
        if time.time() > deadline:
            break
        
        # Remove element at position s (which was factors[s])
        out_f = factors[s]
        freq[out_f] -= 1
        if freq[out_f] == 0:
            distinct_count -= 1
            del freq[out_f]
        
        # Add element at position s+24 (which is factors[s+24])
        in_f = factors[s + 24]
        if freq[in_f] == 0:
            distinct_count += 1
        freq[in_f] += 1
        
        n = 1000 + s
        if distinct_count < best_val:
            best_val = distinct_count
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
