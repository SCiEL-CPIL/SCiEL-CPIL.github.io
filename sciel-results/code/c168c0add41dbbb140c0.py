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
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        p_parts = np.ones(len(multiples), dtype=np.int64)
        temp = multiples.copy()
        while np.any(temp % p == 0):
            mask = temp % p == 0
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts
    
    # Convert to list for faster access in the search loop
    extracted_list = extracted.tolist()
    
    best_n = N_MIN
    best_count = WINDOW + 1  # worst case is 24 distinct
    
    # Sliding window approach
    # Window for n covers indices [n+1, n+24]
    
    # Initialize window for n = N_MIN
    # Indices: N_MIN+1 to N_MIN+24
    counts = {}
    distinct_count = 0
    
    # Add first window elements
    for i in range(1, WINDOW + 1):
        val = extracted_list[N_MIN + i]
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
            distinct_count += 1
            
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = N_MIN
        
    # Slide window
    # Next n is N_MIN + 1
    # Remove index N_MIN + 1 (which is n_prev + 1)
    # Add index N_MIN + 1 + 24 (which is n_new + 24)
    
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        # Remove the element leaving the window: index n
        # The window for n-1 was [n, n+23]
        # The window for n is [n+1, n+24]
        # So we remove extracted[n] and add extracted[n+24]
        
        out_val = extracted_list[n]
        counts[out_val] -= 1
        if counts[out_val] == 0:
            distinct_count -= 1
            
        in_val = extracted_list[n + 24]
        if in_val in counts:
            counts[in_val] += 1
        else:
            counts[in_val] = 1
            distinct_count += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
                
    return best_n
