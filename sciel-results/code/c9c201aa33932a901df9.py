"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    primes_below_24 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes_below_24)
    
    # For each number, we want to know which of these primes divide it
    # We'll compute a bitmask for each number in range [1001, 1000024]
    # where bit i is set if primes_below_24[i] divides the number
    
    # We need numbers from 1001 to 1000024 (since n in [1000,1000000], n+i for i=1..24)
    min_val = 1001
    max_val = 1000000 + 24
    
    # For each prime, create a boolean array of which numbers it divides
    # Then combine into bitmasks
    
    # Use numpy for speed
    size = max_val - min_val + 1
    # bitmask: for each position, which primes divide it
    bitmask = np.zeros(size, dtype=np.int32)
    
    for idx, p in enumerate(primes_below_24):
        # Numbers divisible by p in [min_val, max_val]
        first = ((min_val + p - 1) // p) * p
        if first > max_val:
            continue
        # positions: first, first+p, first+2p, ...
        positions = np.arange(first, max_val + 1, p) - min_val
        bitmask[positions] |= (1 << idx)
    
    # Now slide a window of 24 and count distinct primes
    # For each window starting at position j (corresponding to n = min_val + j - 1),
    # the window covers positions j to j+23
    # We need the OR of bitmasks in the window, then count bits
    
    # Use cumulative approach or sliding window
    # For speed, use a running OR with a count of each bit
    
    best_n = 1000
    best_count = num_primes + 1
    
    # Sliding window: maintain count of each bit position
    bit_counts = np.zeros(num_primes, dtype=np.int32)
    
    # Initialize first window (positions 0 to 23)
    for i in range(24):
        bm = bitmask[i]
        for bit in range(num_primes):
            if bm & (1 << bit):
                bit_counts[bit] += 1
    
    distinct = np.sum(bit_counts > 0)
    if distinct < best_count:
        best_count = distinct
        best_n = min_val - 1  # n such that n+1 = min_val
    
    # Slide
    num_windows = size - 24
    for j in range(1, num_windows):
        # Remove position j-1, add position j+23
        remove_bm = bitmask[j - 1]
        add_bm = bitmask[j + 23]
        
        for bit in range(num_primes):
            if remove_bm & (1 << bit):
                bit_counts[bit] -= 1
            if add_bm & (1 << bit):
                bit_counts[bit] += 1
        
        distinct = np.sum(bit_counts > 0)
        if distinct < best_count:
            best_count = distinct
            best_n = min_val + j - 1  # n such that n+1 = min_val + j
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Ensure best_n is in valid range
    if best_n < 1000:
        best_n = 1000
    if best_n > 1000000:
        best_n = 1000000
    
    return best_n
# EVOLVE-BLOCK-END
