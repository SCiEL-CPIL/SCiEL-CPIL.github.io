import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    min_n = 1000
    max_n = 1000000
    window_size = 24
    
    # We need small_part for indices [min_n+1, max_n+window_size] = [1001, 1000024]
    max_val = max_n + window_size
    min_val = min_n + 1
    
    # Precompute small-prime parts for all numbers up to max_val
    # small_part[i] = product of p^k for each prime p < 24 where p^k || i (full power)
    small_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk <= max_val:
            # For each multiple of pk, multiply by p
            # But we need the FULL power, so we do: for each multiple of pk, multiply by p
            # This gives p^(v_p(n)) in the product
            small_part[pk::pk] *= p
            pk *= p
    
    # Now small_part[i] for i in [min_val, max_val] contains the small-prime part
    # We need to slide a window of size 24 over positions [min_n+1, max_n+24]
    # i.e., windows starting at n where n in [min_n, max_n]
    # Window for n covers indices [n+1, n+24]
    
    # Extract the relevant portion
    # Indices: min_val to max_val, length = max_val - min_val + 1 = 1000024 - 1001 + 1 = 1000024
    sp = small_part[min_val:max_val + 1]
    # sp[0] corresponds to min_val = 1001, sp[k] corresponds to 1001+k
    # Window for n covers sp[n+1 - min_val : n+24 - min_val + 1] = sp[n-999 : n-999+24]
    # Let offset = n - min_n, then window is sp[offset : offset + 24]
    # offset ranges from 0 to max_n - min_n = 999000
    
    # For frequency counting, the max value of small_part is at most max_val ~ 1000024
    max_sp_val = int(np.max(sp))
    freq = np.zeros(max_sp_val + 1, dtype=np.int32)
    
    # Initialize with first window (n = min_n, offset = 0)
    first_window = sp[0:window_size]
    for v in first_window:
        freq[v] += 1
    distinct_count = np.count_nonzero(freq)
    
    best_n = min_n
    best_score = distinct_count
    
    # Slide the window
    # For offset from 1 to max_n - min_n:
    #   Remove sp[offset-1], add sp[offset+23]
    #   Update distinct_count
    
    num_windows = max_n - min_n + 1  # 999001
    
    for offset in range(1, num_windows):
        out_val = int(sp[offset - 1])
        in_val = int(sp[offset + window_size - 1])
        
        freq[out_val] -= 1
        if freq[out_val] == 0:
            distinct_count -= 1
        
        freq[in_val] += 1
        if freq[in_val] == 1:
            distinct_count += 1
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = min_n + offset
        
        if offset % 100000 == 0 and time.time() > deadline:
            break
    
    return best_n
