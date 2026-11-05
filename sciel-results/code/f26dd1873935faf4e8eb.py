import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW
    
    # Precompute extracted[m] = product of p^k for p < 24 where p^k || m
    # Using a faster approach: for each prime, compute p-part via repeated vectorized division
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each power p^k, mark numbers divisible by p^k but not p^(k+1)
        # Actually, simpler: accumulate p-part by multiplying p for each factor
        # Use a different strategy: for each p, create an array of p-parts
        # p-part of m = p^(v_p(m))
        # We can compute this by: start with 1, multiply by p for each division
        
        # More efficient: use the fact that v_p(m) = floor(log_p(m)) adjusted
        # But let's stick with the while loop but optimize it
        
        # Create indices of multiples of p
        multiples = np.arange(p, MAX_VAL + 1, p)
        
        # Compute p-part for these multiples
        p_part = np.ones(len(multiples), dtype=np.int64)
        working = multiples.copy()
        
        # Repeatedly divide by p, tracking how many times each was divided
        while np.any(working % p == 0):
            mask = working % p == 0
            working[mask] //= p
            p_part[mask] *= p
        
        extracted[multiples] *= p_part
    
    # Now vectorized sliding window distinct count
    # Map each extracted value to a compact integer ID
    unique_vals, inverse = np.unique(extracted, return_inverse=True)
    num_unique = len(unique_vals)
    
    # For the sliding window, we need to count distinct IDs in each window
    # Strategy: use a frequency array and slide
    # But to avoid Python loop, we can batch process
    
    # Actually, let's use a different vectorized approach:
    # For each position i in the window start, distinct count = number of unique values in extracted[i:i+WINDOW]
    # We can compute this using a trick with cumulative sums of one-hot encodings,
    # but that's too memory-intensive.
    
    # Alternative: since num_unique is small, we can use a bitset approach
    # But the most practical improvement is to speed up the Python loop
    
    # Let's use a list-based frequency for speed
    extracted_list = inverse.tolist()
    
    best_n = N_MIN
    best_count = WINDOW + 1
    
    # Use a list for freq indexed by compact ID
    freq = [0] * num_unique
    distinct = 0
    
    start_idx = N_MIN + 1
    end_idx = start_idx + WINDOW - 1
    
    for i in range(start_idx, end_idx + 1):
        val = extracted_list[i]
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1
        
    if distinct < best_count:
        best_count = distinct
        best_n = N_MIN
        
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        val_out = extracted_list[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct -= 1
            
        val_in = extracted_list[n + WINDOW]
        if freq[val_in] == 0:
            distinct += 1
        freq[val_in] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 3:
                break
    
    return best_n
