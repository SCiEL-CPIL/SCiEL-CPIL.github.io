import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute smooth parts using sieve-like approach
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            temp = multiple
            while temp % p == 0:
                smooth_part[multiple] *= p
                temp //= p
    
    # Convert to Python int list for faster dict operations
    sp_list = smooth_part.tolist()
    
    # Now find the best n in [1000, 1000000]
    best_n = 1000
    best_count = float('inf')
    
    # Use sliding window with frequency tracking
    # Window: n+1, n+2, ..., n+24
    # We need to track distinct smooth parts in the window
    
    # For efficiency, use a dict to count frequencies
    # Initialize window for n=1000: numbers 1001 to 1024
    freq = {}
    for i in range(1, 25):
        sp = sp_list[1000 + i]
        freq[sp] = freq.get(sp, 0) + 1
    
    current_distinct = len(freq)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, max_n + 1):
        if (n - 1001) % 5000 == 0 and time.time() > deadline:
            break
        
        # Remove n (which was the leftmost of previous window)
        remove_val = sp_list[n]
        cnt = freq[remove_val] - 1
        if cnt == 0:
            del freq[remove_val]
        else:
            freq[remove_val] = cnt
        
        # Add n+24
        add_val = sp_list[n + 24]
        freq[add_val] = freq.get(add_val, 0) + 1
        
        current_distinct = len(freq)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
