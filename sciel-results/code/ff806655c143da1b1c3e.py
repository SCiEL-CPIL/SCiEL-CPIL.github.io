import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Primes strictly below 24
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_NUM_PRIMES = len(_PRIMES)

# Precompute smooth part for a range of values using a sieve-like approach
# We need to compute the 23-smooth part of each number in a window.
# The 23-smooth part of n is the product of all prime factors of n that are < 24,
# with their full multiplicity. Equivalently, divide n by all primes < 24 repeatedly.

def _compute_smooth_parts(start, end):
    """Compute the 23-smooth part for all integers in [start, end]."""
    n_vals = np.arange(start, end + 1, dtype=np.int64)
    smooth = n_vals.copy()
    for p in _PRIMES:
        # For each prime p, divide out all factors of p
        mask = smooth % p == 0
        while np.any(mask):
            smooth[mask] //= p
            mask = smooth[mask] % p == 0
    return smooth

def _score_window(smooth_parts, window_size=24):
    """For each window of window_size consecutive smooth parts, compute the number of distinct values."""
    # smooth_parts has length >= window_size
    # We want, for each i in range(len(smooth_parts) - window_size + 1),
    # the number of distinct values in smooth_parts[i:i+window_size]
    # This is expensive to do naively. Use a sliding window approach.
    # For efficiency with small window (24), we can just loop.
    n = len(smooth_parts)
    best_count = 25  # worst case
    best_offset = 0
    for i in range(n - window_size + 1):
        window = smooth_parts[i:i+window_size]
        distinct = len(set(window.tolist()))
        if distinct < best_count:
            best_count = distinct
            best_offset = i
            if best_count <= 2:
                break
    return best_count, best_offset

def _evaluate_n(n, smooth_parts_cache, cache_start):
    """Evaluate the distinct count for window starting at n."""
    # n corresponds to offset (n - cache_start) in smooth_parts_cache
    offset = n - cache_start
    if offset < 0 or offset + 24 > len(smooth_parts_cache):
        return 25, 0
    window = smooth_parts_cache[offset:offset+24]
    distinct = len(set(window.tolist()))
    return distinct, 0

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Precompute smooth parts for a large range
    # We'll work in chunks to cover [1000, 1000000]
    # Let's compute smooth parts for the entire range at once if possible
    # Range: 1000 to 1000000, that's ~1M numbers. 
    # Computing smooth parts for 1M numbers with 9 primes should be fast with NumPy.
    
    range_start = 1000
    range_end = 1000000
    smooth_parts = _compute_smooth_parts(range_start, range_end)
    
    # Now find the window with minimum distinct count
    # Sliding window of size 24 over smooth_parts
    # For each position i, count distinct in smooth_parts[i:i+24]
    # We need to do this efficiently.
    
    # Strategy: iterate through all windows, tracking distinct count.
    # For a window of size 24, we can use a frequency counter.
    
    best_distinct = 25
    best_n = range_start + 1  # n+i for i=1..24, so n is the base
    
    # Sliding window with frequency tracking
    from collections import Counter
    
    window_size = 24
    n_total = len(smooth_parts)
    
    # Initialize frequency for first window
    freq = Counter(smooth_parts[:window_size].tolist())
    current_distinct = len(freq)
    
    if current_distinct < best_distinct:
        best_distinct = current_distinct
        best_n = range_start
    
    for i in range(1, n_total - window_size + 1):
        # Remove smooth_parts[i-1]
        old_val = smooth_parts[i-1]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        # Add smooth_parts[i+window_size-1]
        new_val = smooth_parts[i+window_size-1]
        freq[new_val] = freq.get(new_val, 0) + 1
        
        current_distinct = len(freq)
        if current_distinct < best_distinct:
            best_distinct = current_distinct
            best_n = range_start + i
            if best_distinct <= 2:
                break
        
        # Time check
        if i % 100000 == 0:
            if time.time() > deadline:
                break
    
    return best_n

# EVOLVE-BLOCK-END
