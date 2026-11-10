"""Optimized program for few distinct small-prime parts in short integer interval."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Max value we need to precompute: n+24 where n <= 1000000
    max_val = 1000024

    # Precompute small-prime factor for all numbers using vectorized sieve
    # sf[x] = product of prime factors p < 24 (with multiplicity)
    sf = np.ones(max_val + 1, dtype=np.int64)

    for p in small_primes:
        pk = p
        while pk <= max_val:
            # Multiply by p for all multiples of pk
            indices = np.arange(pk, max_val + 1, pk)
            sf[indices] *= p
            pk *= p
        if time.time() > deadline - 20:
            break

    window_size = 24
    min_n = 1000
    max_n = 1000000

    best_score = 25
    best_n = min_n

    # Process in chunks to manage memory and allow time checks
    chunk_size = 50000
    
    for chunk_start in range(min_n, max_n + 1, chunk_size):
        if time.time() > deadline - 5:
            break

        chunk_end = min(chunk_start + chunk_size, max_n + 1)
        num_in_chunk = chunk_end - chunk_start
        chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)

        # Get the small-prime factors for window [n+1, n+24]
        # Instead of creating a full 2D index array, extract row by row in vectorized way
        # sf[n+1], sf[n+2], ..., sf[n+24] for all n in chunk
        
        # Create 24 separate 1D arrays, then stack
        cols = np.array([sf[chunk_start + i : chunk_end + i] for i in range(1, 25)])
        # cols has shape (24, num_in_chunk), transpose to (num_in_chunk, 24)
        window_2d = cols.T
        
        # Sort each row
        sorted_rows = np.sort(window_2d, axis=1)
        
        # Count distincts: 1 + number of changes
        diffs = np.diff(sorted_rows, axis=1)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)
        
        # Find minimum in this chunk
        min_count = int(distinct_counts.min())
        if min_count < best_score:
            best_score = min_count
            best_n = int(chunk_n[distinct_counts.argmin()])
            
            if best_score <= 2:
                pass

    return best_n
# EVOLVE-BLOCK-END
