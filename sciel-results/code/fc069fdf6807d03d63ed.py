import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 70  # seconds, safety margin before 90s kill
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 24  # we need up to n+24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Use numpy for faster computation
    smooth_value = np.arange(MAX_VAL + 1, dtype=np.int64)
    
    for p in PRIMES:
        for i in range(p, MAX_VAL + 1, p):
            while smooth_value[i] % p == 0:
                smooth_value[i] //= p
    
    # Search for n in [1000, 1000000] minimizing number of distinct smooth parts in [n+1, n+24]
    # Use sliding window with frequency count for O(1) per step
    best_n = 1000
    best_count = 25  # max possible distinct values in a 24-window
    
    # Initialize frequency map for first window: indices 1001 to 1024
    freq = {}
    for i in range(1001, 1025):
        v = int(smooth_value[i])
        freq[v] = freq.get(v, 0) + 1
    cur_count = len(freq)
    
    if cur_count < best_count:
        best_count = cur_count
        best_n = 1000
    
    # Sliding window: move from n=1000 to n=999999
    # Window for n is [n+1, n+24]
    # When we go from n to n+1, we remove smooth_value[n+1] and add smooth_value[n+25]
    for n in range(1001, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Remove element at position n (was n+1-1 = n... wait)
        # Window for n-1 was [(n-1)+1, (n-1)+24] = [n, n+23]
        # Window for n is [n+1, n+24]
        # So we remove smooth_value[n] and add smooth_value[n+24]
        
        out_val = int(smooth_value[n])
        freq[out_val] -= 1
        if freq[out_val] == 0:
            del freq[out_val]
            cur_count -= 1
        
        in_val = int(smooth_value[n + 24])
        if in_val not in freq:
            freq[in_val] = 1
            cur_count += 1
        else:
            freq[in_val] += 1
        
        if cur_count < best_count:
            best_count = cur_count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1 distinct factor
    
    return best_n
