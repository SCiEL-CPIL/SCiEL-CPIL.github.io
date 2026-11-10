import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000024  # Need up to 1000000 + 24
    
    # Precompute small prime parts
    # small_part[m] is the product of all prime factors of m that are < 24
    small_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        p_power = p
        while p_power <= max_val:
            # Vectorized multiplication
            indices = np.arange(p_power, max_val + 1, p_power)
            small_part[indices] *= p
            p_power *= p
            
    # Check time
    if time.time() > deadline:
        return 1000

    # Sliding window distinct count
    # Window for n is indices n+1 to n+24
    # We start at n = 1000
    # Indices: 1001 to 1024
    
    window_size = 24
    start_n = 1000
    end_n = 1000000
    
    # Initialize frequency map for the first window
    # small_part[start_n + 1 : start_n + 1 + window_size]
    initial_window = small_part[start_n + 1 : start_n + 1 + window_size]
    
    freq = {}
    distinct_count = 0
    for val in initial_window:
        v = int(val)
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct_count += 1
            
    best_count = distinct_count
    best_n = start_n
    
    # Slide the window
    # Next window n+1: remove index (n+1), add index (n+1+window_size)
    # Current window for n: [n+1, ..., n+24]
    # Next window for n+1: [n+2, ..., n+25]
    # So remove small_part[n+1], add small_part[n+25]
    
    for n in range(start_n + 1, end_n + 1):
        if time.time() > deadline:
            break
            
        # Element leaving: index n (which was n+1 in previous window's start? No.)
        # Let's trace carefully.
        # Window for n-1: [n, ..., n+23]
        # Window for n:   [n+1, ..., n+24]
        # Leaving: small_part[n]
        # Entering: small_part[n+24]
        
        old_val = int(small_part[n])
        new_val = int(small_part[n + 24])
        
        # Update old_val
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct_count -= 1
            # Optional: delete key to save memory, but dict size is small
            # del freq[old_val] 
        else:
            # Keep key with 0 count? No, just decrement. 
            # If we don't delete, we need to be careful when adding.
            # It's better to delete to keep freq clean or just let it be 0.
            # If it's 0, and we add it again, we go 0->1.
            pass
            
        # Update new_val
        if new_val in freq:
            if freq[new_val] == 0:
                distinct_count += 1
            freq[new_val] += 1
        else:
            freq[new_val] = 1
            distinct_count += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
