"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_N_SMALL = len(_SMALL_PRIMES)

def _compute_smooth_parts(max_n):
    """
    For each m in [1, max_n], compute the largest divisor of m composed only of primes < 24.
    This is done by removing all factors of small primes from m, then dividing m by the remainder.
    """
    # Start with the number itself
    remainder = np.arange(max_n + 1, dtype=np.int64)
    
    for p in _SMALL_PRIMES:
        for m in range(p, max_n + 1, p):
            while remainder[m] % p == 0:
                remainder[m] //= p
    
    # The smooth part is m // remainder[m]
    smooth = np.ones(max_n + 1, dtype=np.int64)
    smooth[1:] = np.arange(1, max_n + 1, dtype=np.int64) // remainder[1:]
    return smooth

def _find_best_n(small_part, window_size=24, max_n=1000000, deadline=None):
    """
    Find n in [1000, max_n - window_size + 1] such that the number of distinct values in
    small_part[n+1 : n+window_size+1] is minimized.
    
    Note: the window is [n+1, n+24], i.e., 24 consecutive integers starting at n+1.
    """
    best_count = window_size + 1
    best_n = 1000
    
    # We need to scan n from 1000 to max_n - window_size + 1
    # The window for a given n is small_part[n+1 : n+window_size+1]
    # i.e., indices (n+1) to (n+window_size) inclusive
    
    freq = defaultdict(int)
    distinct = 0
    
    # Initialize window at n=1000: indices 1001 to 1024
    start_idx = 1000 + 1
    end_idx = 1000 + window_size
    for i in range(start_idx, end_idx + 1):
        val = small_part[i]
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window: for n from 1001 to max_n - window_size + 1
    # New window: indices (n+1) to (n+window_size)
    # Previous window: indices (n-1+1) to (n-1+window_size) = n to n+window_size-1
    # So we remove index n and add index n+window_size
    
    max_start = max_n - window_size + 1
    for n in range(1001, max_start + 1):
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break
        
        # Remove element at index n (which was the first element of the previous window)
        val_out = small_part[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct -= 1
        
        # Add element at index n + window_size (which is the new last element)
        val_in = small_part[n + window_size]
        if freq[val_in] == 0:
            distinct += 1
        freq[val_in] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin before 90s limit
    
    max_n = 1000000
    window_size = 24
    
    # Precompute small parts
    small_part = _compute_smooth_parts(max_n)
    
    # Find best n
    best_n, best_count = _find_best_n(small_part, window_size, max_n, deadline)
    
    return best_n
# EVOLVE-BLOCK-END
