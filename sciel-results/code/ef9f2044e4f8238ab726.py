"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import Counter

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

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
    
    # Pre-compute small-prime factors for all numbers in [1001, 1000024]
    # Window for n covers n+1..n+24, so we need factors up to 1000000+24
    min_val = 1001
    max_val = 1000000 + 24  # 1000024
    size = max_val - min_val + 1  # 1000024 - 1001 + 1 = 1000024
    
    factors = [1] * size
    for idx in range(size):
        m = min_val + idx
        f = _extract_small_factor(m)
        factors[idx] = f
    
    # Sliding window over n in [1000, 1000000]
    # Window for n: indices for n+1..n+24
    # n=1000 -> factors[0..23] (values 1001..1024)
    # n=1000000 -> factors[999000..999023] (values 1000001..1000024)
    
    best_n = 1000
    best_val = 999999
    
    # Initialize window for n=1000
    freq = Counter()
    for i in range(24):
        freq[factors[i]] += 1
    distinct = len(freq)
    if distinct < best_val:
        best_val = distinct
        best_n = 1000
    
    # Slide window for n=1001 to 1000000
    # n=k -> window starts at index (k - 1000)
    # n=k -> factors[k-1000 .. k-1000+23]
    # n=k+1 -> factors[k-999 .. k-999+23]
    # Remove factors[k-1000], add factors[k-999+23] = factors[k-977]
    
    for k in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Window for n=k: indices [k-1000, k-1000+23]
        # Window for n=k-1: indices [k-1001, k-1001+23]
        # Remove: index k-1001
        # Add: index k-1000+23 = k-977
        
        remove_idx = k - 1001
        add_idx = k - 977
        
        rem_val = factors[remove_idx]
        freq[rem_val] -= 1
        if freq[rem_val] == 0:
            del freq[rem_val]
        
        add_val = factors[add_idx]
        freq[add_val] += 1
        
        distinct = len(freq)
        if distinct < best_val:
            best_val = distinct
            best_n = k
    
    return best_n
# EVOLVE-BLOCK-END
