import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute sf[m] = product of p^k for primes p<24, p^k || m
    sf = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk <= max_val:
            # Multiply sf[m] by p for all m divisible by pk
            # This effectively adds one factor of p for each power
            multiples = np.arange(pk, max_val + 1, pk)
            sf[multiples] *= p
            pk *= p
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if time.time() > deadline - 5:
        return min_n

    # Map sf values to compact indices
    # unique_vals are the distinct small parts
    # inverse[i] is the index of sf[i] in unique_vals
    unique_vals, inverse = np.unique(sf, return_inverse=True)
    num_unique = len(unique_vals)
    
    # Frequency array for the sliding window
    # We only care about the 24 values in the window
    freq = np.zeros(num_unique, dtype=np.int32)
    
    # Initialize window for n = min_n
    # Window: sf[min_n+1] to sf[min_n+24]
    # Indices in inverse: min_n+1 to min_n+24
    init_window = inverse[min_n + 1: min_n + 25]
    for idx in init_window:
        freq[idx] += 1
        
    distinct_count = np.count_nonzero(freq)
    best_count = distinct_count
    best_n = min_n
    
    if best_count == 1:
        return best_n

    # Slide window
    # n goes from min_n + 1 to max_n
    # For current n, window is sf[n+1]..sf[n+24]
    # Previous window (n-1) was sf[n]..sf[n+23]
    # So remove sf[n], add sf[n+24]
    
    # Local variables for speed
    inv = inverse
    f = freq
    bc = best_count
    bn = best_n
    dc = distinct_count
    
    for n in range(min_n + 1, max_n + 1):
        rem_idx = inv[n]
        add_idx = inv[n + 24]
        
        # Remove rem_idx
        if f[rem_idx] == 1:
            f[rem_idx] = 0
            dc -= 1
        else:
            f[rem_idx] -= 1
            
        # Add add_idx
        if f[add_idx] == 0:
            f[add_idx] = 1
            dc += 1
        else:
            f[add_idx] += 1
            
        if dc < bc:
            bc = dc
            bn = n
            if bc == 1:
                return bn
                
        if time.time() > deadline - 2:
            break

    return bn
